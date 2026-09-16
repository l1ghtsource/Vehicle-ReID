import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from hpo.optuna_search_space import suggest_overrides

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("reid_optuna")
CHILDREN: list[subprocess.Popen[bytes]] = []
FINISHED_STATES = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def kill_children() -> None:
    for process in CHILDREN:
        if process.poll() is None:
            process.terminate()
    for process in CHILDREN:
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def run_parallel(
    commands: list[list[str]],
    log_paths: list[Path],
    gpus: list[int],
) -> list[int]:
    processes = []
    handles = []
    CHILDREN.clear()
    try:
        for command, log_path, gpu in zip(commands, log_paths, gpus, strict=True):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            handles.append(handle)
            processes.append(process)
            CHILDREN.append(process)
            LOGGER.info("started gpu=%s pid=%s log=%s", gpu, process.pid, log_path)
        return [process.wait() for process in processes]
    finally:
        CHILDREN.clear()
        for handle in handles:
            handle.close()


def checkpoint_path(fold_dir: Path) -> Path:
    summary = json.loads((fold_dir / "run_summary.json").read_text())
    value = summary["best_checkpoint"] or summary["last_checkpoint"]
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def has_cuda_oom(paths: list[Path]) -> bool:
    markers = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")
    return any(marker in path.read_text(errors="replace") for path in paths for marker in markers)


def weighted_metric(results: list[dict[str, Any]], metric: str) -> float:
    values = np.asarray([result["metrics"][metric] for result in results], dtype=np.float64)
    weights = np.asarray(
        [result["metrics"]["evaluated_queries"] for result in results],
        dtype=np.float64,
    )
    return float(np.average(values, weights=weights))


def save_trial_outputs(trial_dir: Path, metric: str) -> tuple[float, list[dict[str, Any]]]:
    results = []
    frames = []
    embeddings = []
    offset = 0
    for fold in range(5):
        validation_dir = trial_dir / f"fold{fold}" / "val"
        result = json.loads((validation_dir / "metrics.json").read_text())
        frame = pd.read_csv(validation_dir / "oof.csv", dtype={"image_id": str})
        matrix = np.load(validation_dir / "embeddings.npy")
        if len(frame) != len(matrix):
            raise ValueError(f"Fold {fold} OOF metadata and embeddings differ")
        frame["fold"] = fold
        frame["global_embedding_index"] = np.arange(offset, offset + len(frame))
        offset += len(frame)
        results.append(result)
        frames.append(frame)
        embeddings.append(matrix)
    oof = pd.concat(frames, ignore_index=True)
    if oof.image_id.duplicated().any():
        raise ValueError("OOF image IDs are not unique")
    matrix = np.concatenate(embeddings).astype(np.float32)
    np.save(trial_dir / "oof_embeddings.npy", matrix)
    oof.to_csv(trial_dir / "oof.csv", index=False)
    score = weighted_metric(results, metric)
    summary = {
        "metric": metric,
        "oof": score,
        "folds": [
            {
                "fold": int(result["fold"]),
                "value": float(result["metrics"][metric]),
                "evaluated_queries": int(result["metrics"]["evaluated_queries"]),
            }
            for result in results
        ],
    }
    (trial_dir / "score.json").write_text(json.dumps(summary, indent=2))
    return score, results


def run_trial(
    trial: optuna.Trial,
    model: str,
    checkpoint: Path,
    gpus: list[int],
    root: Path,
    metric: str,
    base_overrides: list[str],
) -> float:
    trial_dir = root / f"trial_{trial.number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    sampled = suggest_overrides(trial, model)
    fixed = [
        f"model={model}",
        "model.local_files_only=true",
        f"model.checkpoint_path={checkpoint}",
        "seed=42",
        "checkpoint=null",
        "resume=null",
        "data.n_folds=5",
        "data.num_workers=8",
        "data.pin_memory=true",
        "data.persistent_workers=false",
        "data.prefetch_factor=2",
        "data.batch_size_eval=64",
        "data.verify_files=true",
        "data.validation.query_per_identity=1",
        "data.validation.cross_camera=true",
        "data.validation.exclude_all_same_camera=true",
        "data.validation.ranks=[1,5,10]",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.strategy=auto",
        "trainer.precision=bf16-mixed",
        "trainer.deterministic=true",
        "trainer.benchmark=false",
        "trainer.log_every_n_steps=10",
        "trainer.check_val_every_n_epoch=1",
        "trainer.num_sanity_val_steps=0",
        "trainer.limit_train_batches=1.0",
        "trainer.limit_val_batches=1.0",
        "trainer.max_steps=-1",
        "trainer.enable_progress_bar=false",
        "trainer.sync_batchnorm=false",
        "checkpointing.monitor=val/mAP",
        "checkpointing.mode=max",
        "checkpointing.save_top_k=2",
        "checkpointing.save_last=true",
        "eval.top_k=10",
        "eval.save_distances=false",
        "eval.precision=bf16",
        "postproc.max_dense_gb=4.0",
        "postproc.rerank.device=cpu",
    ]
    overrides = [*base_overrides, *fixed, *sampled]
    (trial_dir / "params.json").write_text(json.dumps(trial.params, indent=2))
    (trial_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))

    train_commands = []
    train_logs = []
    for fold in range(5):
        fold_dir = trial_dir / f"fold{fold}"
        train_commands.append(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "train.py"),
                *overrides,
                f"data.fold={fold}",
                f"output_dir={fold_dir}",
            ]
        )
        train_logs.append(fold_dir / "train.log")
    train_codes = run_parallel(train_commands, train_logs, gpus)
    (trial_dir / "train_exit_codes.json").write_text(json.dumps(train_codes))
    if has_cuda_oom(train_logs):
        raise optuna.TrialPruned("CUDA OOM during training")
    if any(code != 0 for code in train_codes):
        raise RuntimeError(f"Training folds failed: {train_codes}")

    eval_commands = []
    eval_logs = []
    for fold in range(5):
        fold_dir = trial_dir / f"fold{fold}"
        checkpoint_file = checkpoint_path(fold_dir)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
        eval_commands.append(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "eval.py"),
                *overrides,
                f"checkpoint={checkpoint_file}",
                f"data.fold={fold}",
                "eval.split=val",
                "eval.device=cuda",
                f"eval.output_dir={fold_dir / 'val'}",
            ]
        )
        eval_logs.append(fold_dir / "eval.log")
    eval_codes = run_parallel(eval_commands, eval_logs, gpus)
    (trial_dir / "eval_exit_codes.json").write_text(json.dumps(eval_codes))
    if has_cuda_oom(eval_logs):
        raise optuna.TrialPruned("CUDA OOM during evaluation")
    if any(code != 0 for code in eval_codes):
        raise RuntimeError(f"Evaluation folds failed: {eval_codes}")

    score, results = save_trial_outputs(trial_dir, metric)
    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr(
        "fold_values",
        [float(result["metrics"][metric]) for result in results],
    )
    LOGGER.info("trial=%s oof_%s=%.8f", trial.number, metric, score)
    return score


def fail_stale_trials(study: optuna.Study) -> int:
    stale = [trial for trial in study.trials if trial.state == TrialState.RUNNING]
    for trial in stale:
        study.tell(trial.number, state=TrialState.FAIL)
        LOGGER.warning("marked stale trial %s as failed", trial.number)
    return len(stale)


def finished_count(study: optuna.Study) -> int:
    return sum(trial.state in FINISHED_STATES for trial in study.trials)


def write_best(study: optuna.Study, path: Path) -> None:
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not complete:
        LOGGER.warning("no complete trials")
        return
    payload = {
        "number": study.best_trial.number,
        "value": study.best_value,
        "params": study.best_params,
        "trial_dir": study.best_trial.user_attrs.get("trial_dir"),
    }
    path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("best trial=%s value=%.8f", study.best_trial.number, study.best_value)


def prepare_folds(base_overrides: list[str]) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.prepare_folds",
        *base_overrides,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov3_convnext_large")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights/dinov3_large/model.safetensors",
    )
    parser.add_argument("--gpus", default="3,4,5,6,7")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", default="reid_convnext_large")
    parser.add_argument("--metric", default="mAP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 5:
        raise SystemExit("Exactly five GPU IDs are required")
    root = args.out_dir or PROJECT_ROOT / "artifacts" / "optuna" / args.study_name
    storage_path = args.storage or PROJECT_ROOT / "artifacts" / "optuna" / f"{args.study_name}.db"
    log_path = args.log or root / "search.log"
    root.mkdir(parents=True, exist_ok=True)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    prepare_folds(args.override)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path.resolve()}",
        heartbeat_interval=60,
        grace_period=300,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=args.seed, multivariate=True),
        load_if_exists=True,
    )
    stale = fail_stale_trials(study)
    done = finished_count(study)
    remaining = max(0, args.n_trials - done)
    LOGGER.info(
        "study=%s model=%s finished=%s remaining=%s stale=%s",
        args.study_name,
        args.model,
        done,
        remaining,
        stale,
    )
    if remaining == 0:
        write_best(study, root / "best.json")
        return

    def objective(trial: optuna.Trial) -> float:
        return run_trial(
            trial,
            args.model,
            checkpoint,
            gpus,
            root,
            args.metric,
            args.override,
        )

    signal.signal(signal.SIGTERM, lambda *_: (kill_children(), sys.exit(1)))
    try:
        study.optimize(objective, n_trials=remaining, catch=(Exception,))
    except KeyboardInterrupt:
        LOGGER.warning("interrupted")
        kill_children()
        raise SystemExit(130) from None
    finally:
        write_best(study, root / "best.json")


if __name__ == "__main__":
    main()
