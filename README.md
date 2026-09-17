# Vehicle ReID Lab

Vehicle ReID Lab is an experimental open-set vehicle re-identification and image-retrieval pipeline
built with PyTorch, Lightning, and Hydra. It trains an embedding model, evaluates it on
identity-disjoint validation folds, and produces ranked gallery matches for query images.

The project is designed for controlled backbone, loss, augmentation, regularization, and retrieval
experiments. Configuration is explicit, fold manifests are reproducible, and checkpoints are tied
to the dataset and label mapping that produced them.

## Highlights

- Backbone adapters for timm, Hugging Face Transformers, NVIDIA RADIO, and LLM2CLIP/EVA.
- Single-level and multi-level convolutional features.
- GAP, GeM, signed GeM, max, average-max, attention, and CLS pooling.
- ArcFace, CosFace, SphereFace2, triplet, AdaSP, and pytorch-metric-learning losses.
- P × K identity sampling with optional camera-diverse instance selection.
- Backbone freezing, layer-wise learning-rate decay, R-Drop, AWP, and EMA.
- Test-time augmentation over flips, rotations, scales, and bounding-box context.
- Gallery aggregation, average query expansion, k-reciprocal reranking, and GNN reranking.
- Reproducible GroupKFold splits with no vehicle identity overlap between train and validation.
- CPU/offline test suite with 100% line coverage for first-party Python code.

## Requirements

- Linux is recommended.
- Python 3.11 or 3.12.
- `uv` for dependency and environment management.
- A CUDA-compatible GPU for training and large-backbone smoke tests.
- Sufficient storage for datasets, checkpoints, and pretrained weights.

The default development tests do not require a GPU, network access, or production data.

## Installation

Create the environment and install runtime and development dependencies:

```bash
uv sync --extra dev
```

The equivalent Make target is:

```bash
make setup
```

Commands in this README use executables from `.venv`. If the environment is activated, the
`.venv/bin/` prefix can be omitted.

## Dataset layout

The default configuration expects:

```text
data/
├── train.csv
├── test_query.csv
├── test_gallery.csv
└── images/
    ├── image_000001.jpg
    ├── image_000002.jpg
    └── ...
```

Paths can be changed in `configs/config.yaml` or through Hydra overrides:

```bash
.venv/bin/python train.py \
  data.root=/datasets/vehicle-reid \
  data.train_csv=/datasets/vehicle-reid/train.csv \
  data.image_dir=/datasets/vehicle-reid/images
```

### Annotation schema

Every annotation CSV must contain:

- `image_id`: image filename or identifier.
- `x`, `y`: top-left corner of the vehicle bounding box.
- `w`, `h`: positive bounding-box width and height.

`train.csv` must also contain:

- `vehicle_id`: identity used for supervised training and fold construction.

For cross-camera validation, provide:

- `camera_id`: camera identifier. If omitted, it is set to `-1`; cross-camera metrics cannot be
  computed with unknown camera IDs.

Example training rows:

```csv
image_id,vehicle_id,camera_id,x,y,w,h
000001,42,0,31,18,224,136
000002,42,1,12,25,230,140
000003,84,0,48,20,198,132
```

Query and gallery CSV files use the same image and bounding-box fields. Ground-truth
`vehicle_id` values are not required for test retrieval.

If `image_id` has no extension, the loader searches for `.jpg`, `.png`, and `.jpeg`, in that order.
Bounding boxes are clipped to the image boundary. Invalid, empty, non-finite, or duplicated
annotations are rejected instead of being silently corrected.

### External pretraining data

Optional identity-labeled crops live under `extra_data/`. The current loaders are:

```text
extra_data/
├── VeRi/
│   ├── image_train/
│   └── train_label.xml
└── VRIC/
    ├── train_images/
    └── vric_train.txt
```

VeRi uses `image_train/` plus `train_label.xml`. Source:
[VeRi-776 on Kaggle](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)
([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

VRIC uses `train_images/` plus `vric_train.txt` lines of `image identity camera`. Source:
[VRIC](https://qmul-vric.github.io/). The images are derived from [UA-DETRAC](https://detrac-db.rit.albany.edu/).

Both datasets are public academic ReID benchmarks. Their bundled terms are research-only: attribution
is required, redistribution and commercial use are not. That matches the task statement: public
pretrained weights and third-party open datasets are allowed and encouraged, while closed,
proprietary, or unreproducible data are not. List these sources in the solution README when a
pretrained checkpoint is submitted. Do not copy `extra_data/` into the submission; keep the
datasets reproducible from the original downloads.

These images are already cropped, so pretraining does not apply competition bounding boxes.

## Data validation and folds

Audit CSV files, image paths, image readability, and bounding boxes:

```bash
make audit
```

The direct command is:

```bash
.venv/bin/python scripts/audit_data.py --root data --check-images
```

Create persisted identity-disjoint folds:

```bash
make folds
```

Fold construction uses shuffled `GroupKFold` with `vehicle_id` as the group. The generated CSV and
JSON metadata contain the split, data fingerprint, seed, fold count, and protocol version.
Existing manifests are validated before reuse. A stale manifest, changed annotation file, invalid
fold index, or identity leakage causes an error.

Validation query/gallery construction is deterministic. For cross-camera evaluation, a query is
selected from one camera and positive gallery images are taken from other cameras. Identities
without a valid cross-camera positive remain gallery distractors.

## Configuration

The root Hydra configuration is `configs/config.yaml`. Components can be selected independently:

```text
configs/
├── augmentation/
├── experiment/
├── loss/
├── model/
├── optimizer/
└── scheduler/
```

Example composition:

```bash
.venv/bin/python train.py \
  model=resnet50 \
  loss=arcface \
  optimizer=adamw \
  scheduler=cosine \
  data.fold=0
```

Common command-line overrides:

```bash
.venv/bin/python train.py \
  model.head.embedding_dim=768 \
  model.pooling.kind=gem \
  data.image_size='[320,320]' \
  data.sampler.identities=16 \
  data.sampler.instances=4 \
  train.epochs=30 \
  trainer.devices=1
```

Important configuration sections:

- `data`: file paths, image preprocessing, folds, loaders, sampling, and validation protocol.
- `model`: backend, architecture, checkpoint, feature layout, pooling, and embedding head.
- `loss`: one or more weighted classification or metric-learning objectives.
- `optimizer` and `scheduler`: optimizer construction, warmup, and learning-rate schedule.
- `augmentation`: ordered Albumentations and torchvision operations.
- `train`: gradient accumulation, clipping, freezing, R-Drop, AWP, EMA, and layer decay.
- `trainer`: Lightning accelerator, devices, precision, strategy, and batch limits.
- `eval`: checkpoint weights, split, device, precision, TTA, output directory, and top-K.
- `postproc`: query expansion, gallery aggregation, reranking, and dense-memory budget.

## Backbones

Available model presets include:

- `convnext_tiny`
- `resnet50`
- `swin`
- `vit`
- `dinov3_convnext_base`
- `dinov3_convnext_large`
- `dinov3_convnext_multilevel`
- `radio`
- `llm2clip`

Pretrained models are never silently replaced with another architecture. Feature dimensions and
layouts are validated at runtime. Hugging Face adapters require explicit feature dimensions, and
LLM2CLIP uses its required 336 × 336 input resolution.

To test backbone construction and gradients on CUDA:

```bash
.venv/bin/python scripts/check_backbone.py convnext_tiny vit --backward
```

## Losses and sampling

Loss presets are located in `configs/loss/`. The default combined objective uses ArcFace on the
BN-neck feature and hard-mined triplet loss on the raw embedding.

AdaSP requires a P × K sampler with at least two identities and two instances per identity. The
sampler emits contiguous identity groups and can prefer samples from different cameras.

The loss registry rejects unknown objectives, invalid mining modes, non-finite loss values, and
configurations with no positive loss weight.

## Augmentation

`configs/augmentation/reid.yaml` defines the training pipeline. It includes geometric,
photometric, compression, blur, noise, weather, occlusion, erasing, and policy-based transforms.
Each transform has an independent `enabled` switch and probability.

The pipeline does not mix identities. Random state is reseeded per data-loader worker. Misspelled
or unsupported Albumentations arguments are treated as errors.

Image resizing supports:

- `pad`: preserve aspect ratio and pad to the configured canvas.
- `stretch`: resize directly to the configured width and height.

## Training

Run the small smoke configuration:

```bash
make smoke
```

Run a standard fold:

```bash
.venv/bin/python train.py model=convnext_tiny data.fold=0 trainer.devices=1
```

Experiment presets provide larger ready-to-run configurations:

```bash
.venv/bin/python train.py experiment=dino_base data.fold=0
.venv/bin/python train.py experiment=radio data.fold=0
.venv/bin/python train.py experiment=llm2clip data.fold=0
```

For multi-device training, the entrypoint selects a DDP strategy when `trainer.strategy=auto`.
Training uses manual optimization so gradient accumulation, AWP, scheduler updates, and EMA
updates happen in a defined order.

Outputs are written below `output_dir` and include:

- Resolved `config.yaml`.
- Persisted train/query/gallery split CSV files.
- `label_map.json`.
- Lightning logs.
- Last and monitored checkpoints.

### Resume safety

Resume training from a checkpoint:

```bash
.venv/bin/python train.py resume=/path/to/last.ckpt
```

The checkpoint is accepted only if its data fingerprint and label mapping match the current fold.
This prevents accidental continuation on different data or a different identity-to-class mapping.

### External pretraining

`pretrain.py` trains on 100% of one or more extra datasets and validates against 100% of the
competition `train.csv` query/gallery split. Use this to produce a backbone/head checkpoint that
can initialize ordinary competition training.

Train on VeRi, VRIC, or both:

```bash
.venv/bin/python pretrain.py model=dinov3_convnext_base pretrain.datasets=[veri]
.venv/bin/python pretrain.py model=dinov3_convnext_base pretrain.datasets=[vric]
.venv/bin/python pretrain.py model=dinov3_convnext_base pretrain.datasets=[veri,vric]
```

The default config mixes VeRi and VRIC. Identity and camera IDs are remapped so mixed sources do
not collide. Outputs include the usual Lightning checkpoints plus `run_summary.json` with dataset
names, image counts, and checkpoint paths.

Initialize a competition fold from the pretrain checkpoint. Keep the model, pooling, and embedding
head the same. Do not use `resume` for this transfer: pretraining uses a different identity space
and data fingerprint, so `train.py` would reject that checkpoint.

```bash
.venv/bin/python train.py \
  model=dinov3_convnext_base \
  init_checkpoint=runs/pretrain/external_pretrain/<run>/checkpoints/<ckpt>.ckpt \
  data.fold=0
```

`init_checkpoint` loads `model.*` weights only. If the pretrain checkpoint stored EMA weights and
`validation_weights=ema`, those shadows are used; otherwise the raw `state_dict` is used. Losses
and classifiers are created for the competition identity count. `resume` and `init_checkpoint`
cannot be set together.

## Evaluation and retrieval

Evaluate the validation split:

```bash
.venv/bin/python eval.py \
  checkpoint=/path/to/model.ckpt \
  eval.split=val
```

Generate test query/gallery retrieval results:

```bash
.venv/bin/python eval.py \
  checkpoint=/path/to/model.ckpt \
  eval.split=test
```

Select checkpoint weights with:

```bash
eval.weights=auto
eval.weights=raw
eval.weights=ema
```

`auto` uses the validation-weight choice stored in the checkpoint. Requesting EMA weights from a
checkpoint without EMA state is an error.

Evaluation preserves the original query and gallery CSV order. It refuses query/gallery image
overlap because no implicit self-match policy is assumed.

### Test-time augmentation

TTA is configured in `eval.tta`:

```bash
.venv/bin/python eval.py \
  checkpoint=/path/to/model.ckpt \
  eval.tta.enabled=true \
  eval.tta.hflip=true \
  eval.tta.scales='[0.9,1.0,1.1]' \
  eval.tta.rotations='[-5,0,5]'
```

Embeddings from all enabled views and bounding-box context values are averaged and normalized.

### Postprocessing

Enable transductive retrieval postprocessing:

```bash
.venv/bin/python eval.py \
  checkpoint=/path/to/model.ckpt \
  postproc.enabled=true \
  postproc.aqe.enabled=true \
  postproc.rerank.kind=gnn
```

Supported stages:

- Mutual-neighbor gallery prototype aggregation.
- Alpha-weighted average query expansion.
- k-reciprocal reranking.
- GNN reranking.

Dense reranking estimates its memory requirement before allocation and stops when it exceeds
`postproc.max_dense_gb`.

### Evaluation artifacts

The evaluation directory contains:

- `embeddings.npy`: normalized raw model embeddings.
- `retrieval_embeddings.npy`: query and gallery embeddings after enabled expansion stages.
- `distances.npy`: final query-to-gallery distance matrix, when enabled.
- `submission.csv`: stable top-K gallery ranking for each query.
- `query.csv` and `gallery.csv`: exact evaluated row order.
- `embedding_order.csv`: row order corresponding to `embeddings.npy`.
- `metrics.json`: run metadata and validation metrics when labels are available.
- `config.yaml`: resolved evaluation configuration.

Validation reports mAP, mINP, and configured CMC ranks. Queries with no valid positive are counted
and excluded from metric averages; evaluation fails if no query has a valid positive.

## Pretrained weights and offline use

Download supported external weights on a machine with Hugging Face access:

```bash
.venv/bin/python scripts/download_weights.py dinov3_base
.venv/bin/python scripts/download_weights.py dinov3_large
.venv/bin/python scripts/download_weights.py radio
.venv/bin/python scripts/download_weights.py llm2clip
```

Copy the resulting `weights/` directory to the training machine and set the corresponding
`model.checkpoint_path`. Use `model.local_files_only=true` where supported to prevent network
access.

## Cross-validation

Run all five `dino_base` folds in parallel:

```bash
EXPERIMENT=dino_base scripts/train_folds.sh
```

The runner uses a fixed GPU assignment:

- Fold 0: physical CUDA device 3.
- Fold 1: physical CUDA device 4.
- Fold 2: physical CUDA device 5.
- Fold 3: physical CUDA device 6.
- Fold 4: physical CUDA device 7.

Each process sees one GPU through `CUDA_VISIBLE_DEVICES` and runs with `trainer.devices=1`. Existing
Hugging Face weights are loaded from `weights/dinov3_base/model.safetensors` with
`model.local_files_only=true`. Override the checkpoint through `MODEL_CHECKPOINT`. Additional Hydra
overrides can be appended to the command:

```bash
RUN_ROOT=runs/cv/dino_base_v1 \
MODEL_CHECKPOINT=/models/dinov3_base/model.safetensors \
scripts/train_folds.sh \
  data.root=/datasets/vehicle-reid \
  train.epochs=30
```

After every fold finishes, the runner evaluates its best checkpoint on the complete held-out fold,
prints aggregate retrieval metrics, and writes:

- `cv_metrics.json`: mean, standard deviation, and query-weighted mean across folds.
- `oof_embeddings.npy`: normalized embeddings for every training image exactly once.
- `oof.csv`: labels, fold assignments, source rows, and embedding indices.
- `foldN/run_summary.json`: best and last checkpoint paths.
- `foldN/val/metrics.json`: retrieval metrics for one fold.
- Per-fold training and evaluation logs.

To aggregate already completed fold metrics manually:

```bash
.venv/bin/python scripts/aggregate_cv.py \
  runs/fold0/metrics.json \
  runs/fold1/metrics.json \
  runs/fold2/metrics.json \
  --output artifacts/cv_summary.json
```

The report contains per-metric mean, standard deviation, and query-weighted mean. Duplicate fold
IDs are rejected. Embeddings from independently trained folds are not directly comparable and are
therefore not merged by this utility.

## Hyperparameter optimization

The Optuna search runner trains all five folds concurrently, evaluates every best checkpoint, and
maximizes the query-weighted OOF retrieval metric:

```bash
.venv/bin/python -m hpo.run_optuna_search \
  --model dinov3_convnext_large \
  --checkpoint weights/dinov3_large/model.safetensors \
  --gpus 3,4,5,6,7 \
  --study-name convnext_large_all \
  --n-trials 200
```

`hpo/optuna_search_space.py` defines the complete search space. It samples input size and crop
context, P×K sampling, backbone optimization controls, pooling, head, every supported loss and its
parameters, optimizer, scheduler, regularization, every augmentation transform and parameter, TTA,
AQE, gallery aggregation, and reranking. The selected model architecture, checkpoint, five-fold
protocol, data paths, fold assignment, worker and loader settings, precision, deterministic mode,
logging, checkpoint policy, metric protocol, and output paths stay fixed so trials remain
comparable and operational settings do not consume search trials.

Pass data or environment-specific Hydra settings with repeatable `--override` flags:

```bash
.venv/bin/python -m hpo.run_optuna_search \
  --model dinov3_convnext_large \
  --checkpoint /models/dinov3_large/model.safetensors \
  --study-name convnext_large_all \
  --override data.root=/datasets/vehicle-reid \
  --override data.train_csv=/datasets/vehicle-reid/train.csv \
  --override data.image_dir=/datasets/vehicle-reid/images
```

The study is persisted in `artifacts/optuna/<study-name>.db`. Repeating the command with the same
study name and storage resumes it and runs only the missing trials. Interrupted `RUNNING` trials are
marked failed, CUDA OOM trials are pruned, other fold failures are recorded without terminating the
study, and child processes are terminated on interruption. Each trial stores sampled parameters,
exact Hydra overrides, fold logs, exit codes, metrics, OOF metadata, and OOF embeddings. `best.json`
always identifies the best completed trial.

Seed a new study with the completed DINOv3 ConvNeXt Base run. Its configuration and query-weighted
OOF mAP are registered as the first completed Optuna trial without retraining:

```bash
.venv/bin/python -m hpo.run_optuna_search \
  --model dinov3_convnext_base \
  --checkpoint weights/dinov3_base/model.safetensors \
  --gpus 3,4,5,6,7 \
  --study-name convnext_base_all \
  --seed-run runs/cv/dino_base_first \
  --n-trials 200
```

Every trial uses the complete dataset epoch with `data.sampler.steps_per_epoch=null`, and gradient
checkpointing is always disabled. Runtime errors, non-finite runs, and OOM configurations are
recorded as pruned trials instead of failed trials, allowing the study to continue. Fold workers
stop immediately when one fold exits unsuccessfully.

### Optuna Dashboard

Use the local dashboard to monitor the running study, compare completed trials, inspect sampled
parameters, and analyze parameter importance:

```bash
.venv/bin/optuna-dashboard \
  sqlite:///artifacts/optuna/convnext_base_all.db
```

Use `artifacts/optuna/<study-name>.db` for the study you actually started. The file is created by
`hpo.run_optuna_search`; the dashboard will not initialize schema. Pointing it at a missing path
creates an empty SQLite file and then fails with `no such table: version_info`.

The dashboard is available at `http://127.0.0.1:8080`. For a remote machine, either forward port
8080 over SSH or expose the service on the machine's network interface:

```bash
.venv/bin/optuna-dashboard \
  --host 0.0.0.0 \
  --port 8080 \
  sqlite:///artifacts/optuna/convnext_base_all.db
```

The dashboard can safely read the SQLite study while the search runner is writing new trials.
Per-epoch fold metrics remain available in
`artifacts/optuna/<study-name>/trial_<number>/fold<fold>/logs/version_0/metrics.csv`.

## Quality gates

Format first-party code, then run Ruff and ty:

```bash
make lint
```

The equivalent commands are:

```bash
.venv/bin/ruff format .
.venv/bin/ruff check .
.venv/bin/ty check
```

Run tests:

```bash
make test
```

The default pytest command enforces 100% line coverage over all first-party runtime code. Tests use
synthetic data and mocks for network, GPU-only, and large-model operations.

Ruff and ty check first-party source and tests. Vendored code under `third_party/` is the only code
exclusion. Function-local imports are forbidden by Ruff rule `PLC0415`.

## Reproducibility and safety

- Python and core ML dependency versions are pinned in `pyproject.toml` and `uv.lock`.
- Fold generation is deterministic for a fixed annotation file, seed, and fold count.
- Data fingerprints are persisted and checked when folds or checkpoints are reused.
- Validation identities are disjoint from training identities.
- Partial validation runs are treated as smoke checks and do not report retrieval scores.
- Non-finite embeddings, losses, and reranking results fail explicitly.
- Model adapters validate returned feature count, shape, and configured dimensions.

## Project layout

```text
augmentations/  Config-driven image augmentation pipeline
configs/        Hydra model, loss, optimizer, scheduler, and experiment presets
dataset/        Annotation validation, folds, datasets, data module, pretrain loaders, and samplers
extra_data/     Optional external identity-labeled crops for pretraining
models/         Backbone adapters, pooling layers, and embedding model
modules/        Lightning module, losses, metrics, inference, optimization, and regularization
postproc/       Retrieval expansion, aggregation, and reranking
scripts/        Dataset audit, fold creation, weight download, model checks, and CV aggregation
tests/          CPU/offline unit, integration, configuration, and entrypoint tests
third_party/    Vendored upstream implementations
pretrain.py     Hydra pretraining entrypoint on extra data
train.py        Hydra training entrypoint
eval.py         Checkpoint-driven evaluation and retrieval entrypoint
```

## Current limitations

- Open-set refusal is not implemented; every query currently receives a ranked gallery result.
- Full production-scale DDP, GPU memory, and throughput behavior must be validated in the target
  training environment.
- Transductive postprocessing uses the complete evaluation query/gallery set and must be disabled
  when that evaluation protocol is not allowed.
- Foundation-model checkpoints can require substantial disk space and GPU memory.
