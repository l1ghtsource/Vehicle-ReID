import argparse
import json
import runpy
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import TrialState

import hpo.optuna_search_space as search_space
import hpo.run_optuna_search as runner


class FakeTrial:
    def __init__(self, choices=None, number=0):
        self.choices = choices or {}
        self.number = number
        self.params = {}
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        value = self.choices.get(name, choices[0])
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, log=False):
        value = self.choices.get(name, (low * high) ** 0.5 if log else (low + high) / 2)
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high):
        value = int(self.choices.get(name, low))
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


@pytest.mark.parametrize("loss", search_space.LOSS_PRESETS)
def test_search_space_all_losses(loss):
    trial = cast(
        optuna.Trial,
        FakeTrial(
            {
                "loss": loss,
                "optimizer": "adamw",
                "train.ema.enabled": True,
                "eval.weights": "ema",
            }
        ),
    )
    overrides = search_space.suggest_overrides(trial, "dinov3_convnext_large")
    assert f"loss={loss}" in overrides
    assert "eval.weights=ema" in overrides


@pytest.mark.parametrize("optimizer", ["adamw", "sgd", "lamb", "lion"])
@pytest.mark.parametrize("model", ["vit", "radio", "llm2clip", "dinov3_convnext_large"])
def test_search_space_models_optimizers_and_conditions(model, optimizer):
    trial = cast(
        optuna.Trial,
        FakeTrial(
            {
                "loss": "arcface_adasp",
                "optimizer": optimizer,
                "train.awp.enabled": True,
            }
        ),
    )
    overrides = search_space.suggest_overrides(trial, model)
    assert f"optimizer={optimizer}" in overrides
    assert "train.accumulate_grad_batches=1" in overrides
    if model == "llm2clip":
        assert "data.image_size=[336,336]" in overrides


def test_search_space_helpers():
    trial = cast(optuna.Trial, FakeTrial())
    assert search_space._value(True) == "true"
    assert search_space._value("raw") == "raw"
    assert search_space._sample(trial, "flag", ("bool",)) is False
    assert search_space._sample(trial, "choice", ("categorical", ["a"])) == "a"
    assert search_space._sample(trial, "integer", ("int", 1, 2)) == 1
    assert search_space._sample(trial, "float", ("float", 1.0, 2.0)) == 1.5
    with pytest.raises(ValueError, match="Unknown"):
        search_space._sample(trial, "bad", ("bad",))
    output = []
    search_space._append(output, "x", [1, 2])
    assert output == ["x=[1,2]"]


def create_fold_outputs(root, duplicate=False, mismatch=False):
    for fold in range(5):
        directory = root / f"fold{fold}" / "val"
        directory.mkdir(parents=True)
        image_id = "same" if duplicate else f"image_{fold}"
        pd.DataFrame([{"embedding_index": 0, "image_id": image_id, "vehicle_id": fold, "fold": fold}]).to_csv(
            directory / "oof.csv", index=False
        )
        rows = 2 if mismatch and fold == 0 else 1
        np.save(directory / "embeddings.npy", np.ones((rows, 3), dtype=np.float32))
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "metrics": {"mAP": 0.5 + fold / 10, "evaluated_queries": fold + 1},
                }
            )
        )


def test_metric_and_trial_output_aggregation(tmp_path):
    create_fold_outputs(tmp_path)
    score, results = runner.save_trial_outputs(tmp_path, "mAP")
    expected = np.average([0.5, 0.6, 0.7, 0.8, 0.9], weights=[1, 2, 3, 4, 5])
    assert score == pytest.approx(expected)
    assert len(results) == 5
    assert np.load(tmp_path / "oof_embeddings.npy").shape == (5, 3)
    assert runner.weighted_metric(results, "mAP") == pytest.approx(expected)

    mismatch = tmp_path / "mismatch"
    create_fold_outputs(mismatch, mismatch=True)
    with pytest.raises(ValueError, match="differ"):
        runner.save_trial_outputs(mismatch, "mAP")

    duplicate = tmp_path / "duplicate"
    create_fold_outputs(duplicate, duplicate=True)
    with pytest.raises(ValueError, match="unique"):
        runner.save_trial_outputs(duplicate, "mAP")


def test_checkpoint_and_oom_helpers(tmp_path, monkeypatch):
    fold = tmp_path / "fold"
    fold.mkdir()
    relative = tmp_path / "relative.ckpt"
    relative.touch()
    (fold / "run_summary.json").write_text(
        json.dumps({"best_checkpoint": str(relative), "last_checkpoint": ""})
    )
    assert runner.checkpoint_path(fold) == relative
    absolute = tmp_path / "absolute.ckpt"
    (fold / "run_summary.json").write_text(
        json.dumps({"best_checkpoint": "", "last_checkpoint": str(absolute.resolve())})
    )
    assert runner.checkpoint_path(fold) == absolute.resolve()

    log = tmp_path / "run.log"
    log.write_text("ok")
    assert not runner.has_cuda_oom([log])
    log.write_text("CUDA out of memory")
    assert runner.has_cuda_oom([log])
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    (fold / "run_summary.json").write_text(
        json.dumps({"best_checkpoint": "relative.ckpt", "last_checkpoint": ""})
    )
    assert runner.checkpoint_path(fold) == tmp_path / "relative.ckpt"


def test_run_parallel_real_processes(tmp_path):
    commands = [
        [sys.executable, "-c", "print('one')"],
        [sys.executable, "-c", "print('two')"],
    ]
    logs = [tmp_path / "one.log", tmp_path / "two.log"]
    assert runner.run_parallel(commands, logs, [3, 4]) == [0, 0]
    assert logs[0].read_text().strip() == "one"
    assert runner.CHILDREN == []


class KillProcess:
    def __init__(self, timeout=False):
        self.timeout = timeout
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired("x", float(timeout or 0))
        return 0

    def kill(self):
        self.killed = True


def test_kill_children():
    first = KillProcess()
    second = KillProcess(timeout=True)
    runner.CHILDREN[:] = cast(list[subprocess.Popen[bytes]], [first, second])
    runner.kill_children()
    assert first.terminated
    assert second.killed
    runner.CHILDREN.clear()


def test_run_trial_success_and_failures(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    trial = cast(optuna.Trial, FakeTrial())
    monkeypatch.setattr(runner, "suggest_overrides", lambda trial, model: ["train.epochs=1"])
    monkeypatch.setattr(runner, "checkpoint_path", lambda fold_dir: checkpoint)
    monkeypatch.setattr(
        runner,
        "save_trial_outputs",
        lambda directory, metric: (
            0.75,
            [{"metrics": {metric: 0.75}} for _ in range(5)],
        ),
    )

    monkeypatch.setattr(runner, "run_parallel", lambda commands, logs, gpus: [0] * 5)
    monkeypatch.setattr(runner, "has_cuda_oom", lambda logs: False)
    assert runner.run_trial(trial, "model", checkpoint, [3, 4, 5, 6, 7], tmp_path, "mAP", []) == 0.75
    overrides = json.loads((tmp_path / "trial_00000" / "overrides.json").read_text())
    assert "data.num_workers=8" in overrides
    assert "trainer.precision=bf16-mixed" in overrides
    assert "checkpointing.monitor=val/mAP" in overrides

    monkeypatch.setattr(runner, "has_cuda_oom", lambda logs: True)
    with pytest.raises(optuna.TrialPruned, match="training"):
        runner.run_trial(trial, "model", checkpoint, [3, 4, 5, 6, 7], tmp_path, "mAP", [])

    monkeypatch.setattr(runner, "has_cuda_oom", lambda logs: False)
    monkeypatch.setattr(runner, "run_parallel", lambda commands, logs, gpus: [1, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="Training"):
        runner.run_trial(trial, "model", checkpoint, [3, 4, 5, 6, 7], tmp_path, "mAP", [])

    calls = 0

    def eval_failure(commands, logs, gpus):
        nonlocal calls
        calls += 1
        return [0] * 5 if calls == 1 else [1, 0, 0, 0, 0]

    monkeypatch.setattr(runner, "run_parallel", eval_failure)
    with pytest.raises(RuntimeError, match="Evaluation"):
        runner.run_trial(trial, "model", checkpoint, [3, 4, 5, 6, 7], tmp_path, "mAP", [])

    calls = 0
    oom_calls = 0

    def eval_oom(commands, logs, gpus):
        nonlocal calls
        calls += 1
        return [0] * 5

    def second_oom(logs):
        nonlocal oom_calls
        oom_calls += 1
        return oom_calls == 2

    monkeypatch.setattr(runner, "run_parallel", eval_oom)
    monkeypatch.setattr(runner, "has_cuda_oom", second_oom)
    with pytest.raises(optuna.TrialPruned, match="evaluation"):
        runner.run_trial(trial, "model", checkpoint, [3, 4, 5, 6, 7], tmp_path, "mAP", [])

    missing = tmp_path / "missing.ckpt"
    monkeypatch.setattr(runner, "run_parallel", lambda commands, logs, gpus: [0] * 5)
    monkeypatch.setattr(runner, "has_cuda_oom", lambda logs: False)
    monkeypatch.setattr(runner, "checkpoint_path", lambda fold_dir: missing)
    with pytest.raises(FileNotFoundError):
        runner.run_trial(trial, "model", checkpoint, [3, 4, 5, 6, 7], tmp_path, "mAP", [])


def test_study_helpers_and_best(tmp_path):
    study = optuna.create_study(direction="maximize")
    running = study.ask()
    assert runner.fail_stale_trials(study) == 1
    assert study.trials[running.number].state == TrialState.FAIL
    trial = study.ask()
    study.tell(trial, 0.8)
    assert runner.finished_count(study) == 2
    path = tmp_path / "best.json"
    runner.write_best(study, path)
    assert json.loads(path.read_text())["value"] == 0.8

    empty = optuna.create_study(direction="maximize")
    runner.write_best(empty, tmp_path / "empty.json")
    assert not (tmp_path / "empty.json").exists()


def test_prepare_folds_and_parse_args(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, cwd, check: calls.append(command))
    runner.prepare_folds(["data.root=x"])
    assert "scripts.prepare_folds" in calls[0]
    monkeypatch.setattr(sys, "argv", ["run_optuna_search", "--gpus", "0,1,2,3,4"])
    args = runner.parse_args()
    assert args.model == "dinov3_convnext_large"


class MainStudy:
    def __init__(self, interrupt=False, complete=False):
        self.interrupt = interrupt
        self.trials = []
        self.best_trial = SimpleNamespace(number=1, user_attrs={"trial_dir": "x"})
        self.best_value = 0.9
        self.best_params = {"x": 1}
        if complete:
            self.trials = [SimpleNamespace(state=TrialState.COMPLETE)]

    def optimize(self, objective, n_trials, catch):
        if self.interrupt:
            raise KeyboardInterrupt
        objective(cast(optuna.Trial, FakeTrial()))


def args_for_main(tmp_path, **kwargs):
    values: dict[str, Any] = {
        "model": "dinov3_convnext_large",
        "checkpoint": tmp_path / "model.safetensors",
        "gpus": "3,4,5,6,7",
        "n_trials": 1,
        "study_name": "test",
        "metric": "mAP",
        "seed": 42,
        "storage": tmp_path / "study.db",
        "out_dir": tmp_path / "study",
        "log": tmp_path / "study.log",
        "override": [],
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_main_paths(tmp_path, monkeypatch):
    args = args_for_main(tmp_path)
    args.checkpoint.touch()
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "prepare_folds", lambda overrides: None)
    monkeypatch.setattr(runner, "setup_logging", lambda path: None)
    monkeypatch.setattr(runner.optuna.storages, "RDBStorage", lambda **kwargs: object())
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner, "run_trial", lambda *args: 0.7)
    monkeypatch.setattr(runner, "write_best", lambda study, path: path.touch())

    study = MainStudy()
    monkeypatch.setattr(runner.optuna, "create_study", lambda **kwargs: study)
    runner.main()
    assert (args.out_dir / "best.json").exists()

    complete = MainStudy(complete=True)
    monkeypatch.setattr(runner.optuna, "create_study", lambda **kwargs: complete)
    runner.main()

    interrupted = MainStudy(interrupt=True)
    monkeypatch.setattr(runner.optuna, "create_study", lambda **kwargs: interrupted)
    monkeypatch.setattr(runner, "kill_children", lambda: None)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 130

    args.gpus = "3"
    with pytest.raises(SystemExit, match="five"):
        runner.main()
    args.gpus = "3,4,5,6,7"
    args.checkpoint = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        runner.main()


def test_module_entrypoint(tmp_path, monkeypatch):
    args = args_for_main(tmp_path, n_trials=0)
    args.checkpoint.touch()
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(optuna.storages, "RDBStorage", lambda **kwargs: object())
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: MainStudy())
    runpy.run_module("hpo.run_optuna_search", run_name="__main__")
