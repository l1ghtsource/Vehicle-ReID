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
- Test-time augmentation over flips, rotations, scales, and bounding-box context of a single image.
- Streaming retrieval postproc: gallery-side aggregation and per-query rerank. AQE is OOF-only.
- Open-set refusal: cosine threshold, CatBoost, and TabM on retrieval-set features.
- EfficientLoFTR pair matching as a retrieval visualizer and top-k verifier.
- Reproducible GroupKFold splits with no vehicle identity overlap between train and validation.
- CPU/offline test suite with 100% line coverage for first-party Python code.
- Contest serving file is a compact EMA `.pt` exported from a Lightning checkpoint.

## Requirements

- Linux is recommended.
- Python 3.11 or 3.12.
- `uv` for dependency and environment management.
- Git LFS for `weights/finetuned/` serving checkpoints.
- A CUDA-compatible GPU for training and large-backbone smoke tests.
- Sufficient storage for datasets, checkpoints, and pretrained weights.

The default development tests do not require a GPU, network access, or production data.

## Installation

Create the environment and install runtime, development, and notebook dependencies:

```bash
uv sync --extra dev
```

The equivalent Make target is:

```bash
make setup
```

After changing dependencies, refresh the lock and the hashed Docker freeze:

```bash
make lock
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
├── refusal/
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
- `postproc`: gallery aggregation, per-query reranking, dense-memory budget. AQE is not used at serve.

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
.venv/bin/python train.py experiment=current_best_tuned data.fold=0
```

`current_best_tuned` is LLM2CLIP EVA02-L-14-336 trained with the Optuna `convnext_base_all` trial 23
recipe (5-fold OOF mAP 0.847, mAP@10 0.834): 336 input, `local_parts=0`, PK 16×2, ArcFace+AdaSP,
linear schedule, EMA. Pass `model=` to reuse the recipe with another backbone.

![Per-fold OOF metrics for current_best_tuned](notebooks/eva02/readme_figs/current_best_folds.png)

*Identity-disjoint 5-fold OOF on EVA02 (`runs/cv/eva02_trial23`). Query-weighted means: mAP 0.847,
mAP@10 0.834, Rank-1 0.842. Fold 4 is the strongest; fold 3 is the weakest. Checkpoint selection
and Optuna still use full-gallery mAP; mAP@10 is the contest ranking metric.*

![Fold-0 retrieval examples: query, Rank-1, first true positive](notebooks/eva02/readme_figs/current_best_pairs.jpg)

*Fold-0 examples. Green Rank-1 is a correct identity; red is a lookalike. The third column is the
first true positive in the ranking. Hard misses are usually the same body style and color, not a
random vehicle. Full case study: `notebooks/eva02/oof_analysis.ipynb`.*

### Full retrain

`data.full_retrain=true` trains on every competition identity (all five fold groups). There is no
held-out val split and no `val/mAP` checkpoint monitor: Lightning saves `last.ckpt` after a fixed
epoch budget. `eval.split=val` refuses that checkpoint, because train IDs cover the held-out fold.

Set the budget to the rounded mean of the five best-checkpoint epochs from an identity-disjoint CV
run. The filenames are 0-based Lightning epochs (`epoch025.ckpt` → 25). For
`runs/cv/eva02_trial23` that is `(25+25+24+21+12)/5 = 21.4 → 21`, so `train.epochs=21` (epochs 0–20).
Pass `data.cv_dir=` to compute that mean inside `train.py`; otherwise keep `train.epochs` as in the
experiment YAML.

```bash
.venv/bin/python train.py \
  experiment=current_best_tuned \
  data.full_retrain=true \
  data.cv_dir=runs/cv/eva02_trial23 \
  output_dir=runs/full/eva02_trial23
```

Export the last EMA weights for contest serving. This is the default `scripts/export_serving.py`
source when `runs/full/eva02_trial23/run_summary.json` exists:

```bash
.venv/bin/python scripts/export_serving.py \
  --checkpoint runs/full/eva02_trial23/checkpoints/last.ckpt \
  --output weights/finetuned/eva02.pt \
  --weights ema \
  --sha256
```

Refit CatBoost in that same embedding space. `--full-retrain` re-embeds the five CV query/gallery
splits with the serving checkpoint (all identities, one model) and fits the same 200/4/0.08 recipe
on every 50/50 pack. The accept threshold stays the nested 5-fold inner-CV value `0.6719` in
`configs/refusal/`; full-retrain data are in-sample and must not retune the operating point.
`eval.py` cannot build this pack: the full-retrain checkpoint contains every identity, so
`eval.split=val` is rejected.

```bash
.venv/bin/python scripts/export_refusal.py \
  --full-retrain \
  --cv runs/cv/eva02_trial23 \
  --checkpoint runs/full/eva02_trial23/checkpoints/last.ckpt \
  --device cuda:1 \
  --output weights/finetuned/eva02_catboost.cbm \
  --sha256
```

Omit `--full-retrain` to keep the older 5-fold OOF CatBoost.

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

Train on VeRi, VRIC, or both. `scripts/pretrain.sh` selects the GPU, Hydra model group, extra
datasets, and experiment recipe. It defaults to `experiment=current_best_tuned` and local
`weights/` checkpoints:

```bash
scripts/pretrain.sh cuda:2 dinov3_convnext_base veri
scripts/pretrain.sh cuda:2 llm2clip vric
scripts/pretrain.sh 2 dinov3_convnext_base veri,vric
scripts/pretrain.sh cuda:2 radio both smoke
```

The first argument is `cuda:N` or `N`. The second is the Hydra model group. The third is `veri`,
`vric`, `veri,vric`, or `both`. An optional fourth token without `=` is the experiment name;
otherwise extra tokens are Hydra overrides. LLM2CLIP is forced to 336 × 336 and `local_parts=0`.
Set `EXPERIMENT`, `MODEL_CHECKPOINT`, `NAME`, or `PYTHON` to override the defaults.

Direct Hydra remains available:

```bash
.venv/bin/python pretrain.py experiment=current_best_tuned model=llm2clip pretrain.datasets=[veri]
.venv/bin/python pretrain.py experiment=current_best_tuned model=llm2clip pretrain.datasets=[vric]
```

The default config mixes VeRi and VRIC. Identity and camera IDs are remapped so mixed sources do
not collide. Outputs include the usual Lightning checkpoints plus `run_summary.json` with dataset
names, image counts, and checkpoint paths.

Initialize a competition fold from the pretrain checkpoint. Keep the model, pooling, and embedding
head the same. Do not use `resume` for this transfer: pretraining uses a different identity space
and data fingerprint, so `train.py` would reject that checkpoint.

```bash
.venv/bin/python train.py \
  experiment=current_best_tuned \
  init_checkpoint=runs/pretrain/llm2clip_veri/<run>/checkpoints/<ckpt>.ckpt \
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
  checkpoint=weights/finetuned/eva02.pt \
  eval.split=val
```

Generate test query/gallery retrieval results:

```bash
.venv/bin/python eval.py \
  checkpoint=weights/finetuned/eva02.pt \
  eval.split=test
```

Open-set refusal is applied only to contest `candidates.csv`. Serving heads and frozen thresholds
live in `configs/refusal/`. Default local `refusal=none` writes every query. The contest Docker
image defaults to `refusal=eva02_ensemble`. EVA02 serving presets drop refused queries: **no rows**
for that `query_id` (not an empty `gallery_id`, not a sentinel). `submission.csv` is ranking-only:
exactly `eval.top_k` (10) gallery IDs for **every** query, including open-set and refused ones, with
no score column and no skipped `query_id`.

```bash
.venv/bin/python eval.py checkpoint=weights/finetuned/eva02.pt eval.split=test refusal=eva02_threshold
.venv/bin/python eval.py checkpoint=weights/finetuned/eva02.pt eval.split=test refusal=eva02_model
.venv/bin/python eval.py checkpoint=weights/finetuned/eva02.pt eval.split=test refusal=eva02_ensemble
```

`eva02_model` and `eva02_ensemble` need a `.cbm` at `refusal.model_path` (default
`weights/finetuned/eva02_catboost.cbm`). Export it from EVA02 OOF embeddings with
`scripts/export_refusal.py`. Thresholds in those YAMLs are frozen EVA02 nested 5-fold inner-CV
maxima of `0.7 × F1 + 0.3 × TNR`, not retuned on the test set. TabM remains a
`refusal/` calibration head and is not a submit preset.

Select checkpoint weights with:

```bash
eval.weights=auto
eval.weights=raw
eval.weights=ema
```

`auto` uses the weight kind stored in the file (`validation_weights` on a Lightning `.ckpt`,
`weights` on a serving `.pt`). Requesting EMA weights from a checkpoint without EMA state is an
error. A serving `.pt` contains one exported set (usually EMA); asking for the other kind is an
error.

Evaluation preserves the original query and gallery CSV order. It refuses query/gallery image
overlap because no implicit self-match policy is assumed.

### Zero-shot pretrained probe

`scripts/zero_shot.py` scores a Hydra model config on 100% of `train.csv` without training. The
query/gallery split is the same protocol `pretrain.py` uses for extra-data validation: one query per
identity when a cross-camera positive exists, remaining images as gallery. Random projection,
BN-neck, and attention pooling are disabled so the pretrained backbone features are used directly.
GAP pooling is the default; Hydra overrides still apply after those defaults.

Probe DINOv3 ConvNeXt Base from local weights:

```bash
.venv/bin/python scripts/zero_shot.py dinov3_convnext_base \
  model.checkpoint_path=weights/dinov3_base/model.safetensors \
  model.local_files_only=true \
  eval.device=cuda:2
```

Run every backbone in `weights/` sequentially on one device. The device is the first argument
(`cuda:2`, `cuda:0`, and so on). Per-model metrics go to `artifacts/zero_shot/<model>/metrics.json`,
and a combined file is written to `artifacts/zero_shot/summary.json`:

```bash
scripts/zero_shot_weights.sh cuda:2
```

Measured out-of-the-box retrieval on 100% of competition `train.csv` (1541 queries, 5550 gallery
images, cross-camera protocol, no TTA or reranking). Junk is same `vehicle_id` and `camera_id`;
other identities on the query camera stay in the gallery. These scores are a pretrained-backbone
probe, not identity-disjoint fold OOF. `mAP` is full-gallery average precision; `mAP@10` is the
official submission metric (top-10, denominator `min(n_pos, 10)`).

| Model | mAP | mAP@10 | Rank-1 | Rank-5 | Rank-10 | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3 ConvNeXt Base | 0.168 | 0.141 | 0.162 | 0.286 | 0.356 | 0.114 |
| DINOv3 ConvNeXt Large | 0.182 | 0.152 | 0.167 | 0.310 | 0.399 | 0.132 |
| RADIO C-RADIOv4-SO400M | 0.148 | 0.123 | 0.147 | 0.263 | 0.319 | 0.098 |
| LLM2CLIP EVA02-L-14-336 | 0.301 | 0.268 | 0.313 | 0.471 | 0.580 | 0.212 |

### Test-time augmentation

TTA is configured in `eval.tta`:

```bash
.venv/bin/python eval.py \
  checkpoint=weights/finetuned/eva02.pt \
  eval.tta.enabled=true \
  eval.tta.hflip=true \
  eval.tta.scales='[0.9,1.0,1.1]' \
  eval.tta.rotations='[-5,0,5]'
```

Embeddings from all enabled views and bounding-box context values are averaged and normalized.

### Embedding interpretation

`interp/` attributes a retrieval embedding, not a classifier logit. The default target is
pre-normalization embedding energy; pass a gallery vector as `reference` to explain
`cosine(query, gallery)` instead. `interpret_pair` embeds both images, then attributes each
side of that cosine.

| Method | What it shows |
| --- | --- |
| `pooling` | Spatial weights of the trained attention pooler (the aggregation the embedding actually uses) |
| `last_attn` | Last-layer CLS-to-patch attention, averaged over heads |
| `rollout` | Attention rollout with residual identity (Abnar & Zuidema, 2020); image-general, not pair-specific |
| `grad_rollout` | Same residual rollout, but attention is weighted by the cosine-similarity gradient |
| `chefer` | Transformer attribution: attention × gradient relevancy (Chefer et al., 2021) |
| `gradsim` | `|grad × activation|` on last patch tokens with cosine (or energy) as the scalar target |
| `gradcam` | Grad-CAM on the last backbone feature map (Selvaraju et al., 2017) |
| `hirescam` | HiResCAM, element-wise gradient × activation |
| `layercam` | LayerCAM, ReLU(gradient) × activation |
| `eigencam` | EigenCAM, first principal component of activations (no labels/gradients) |
| `occlusion` | Cosine (or energy) drop after zeroing a patch block; `block=` groups the backbone grid |

```python
from interp import interpret, interpret_pair, overlay

maps = interpret(model, images, method="gradcam")
maps = interpret(model, images, method="chefer", reference=gallery_embedding)
query_maps, gallery_maps, cosine = interpret_pair(
    model, query, gallery, method="gradsim", block=4
)
canvas = overlay(crop_hwc, query_maps[0])
```

![Image-level attribution methods on the same OOF queries](notebooks/eva02/readme_figs/interp_methods.jpg)

*Each row is one fold-0 OOF query. `pooling` is the trained spatial attention that actually builds
the embedding. `last_attn` / `rollout` are CLS maps (this recipe does not retrieve with CLS). CAM
methods here use embedding energy (no gallery vector). Occlusion is skipped in this grid because
EVA02's 24×24 patch grid is too many forwards.*

![Pair cosine attribution: query and gallery maps for gradsim, grad-rollout, occlusion, Chefer](notebooks/eva02/readme_figs/interp_pairs.jpg)

*Pair maps: `interpret_pair` backprops `cosine(z_q, z_g)` into both crops. Columns are query and
gallery for `gradsim`, `grad_rollout`, `occlusion` (`block=4`), and `chefer`. Rows alternate true
positive vs Rank-1. Warm regions are the patches that currently support that cosine. Notebook:
`notebooks/eva02/interp.ipynb`.*

### Pair matching (EfficientLoFTR)

`matching/` runs detector-free semi-dense correspondence on a query/gallery crop pair using the
local Hugging Face checkpoint `weights/efficientloftr`. The embedding ranker still decides the
shortlist. Matching is a second opinion on that shortlist: how many keypoints correspond, how
confident those correspondences are, and how many survive a homography RANSAC.

| Output | Meaning |
| --- | --- |
| `n_matches` | Correspondences above the matcher threshold (default 0.2) |
| `score_sum` / `score_mean` | Sum / mean of those correspondence scores |
| `n_inliers` / `inlier_ratio` | RANSAC homography inliers; a geometric check that the matches are consistent |

`reorder_head` re-sorts only the cosine top-`k`. `hybrid_head` min-max mixes cosine and a match
statistic inside that head. The matcher is not applied to the full gallery.

```python
from matching import draw_matches, load_matcher, match_pair, match_stats, reorder_head

processor, matcher = load_matcher("weights/efficientloftr", device="cuda:2")
pair = match_pair(processor, matcher, query_crop, gallery_crop, threshold=0.2)
stats = match_stats(pair["keypoints0"], pair["keypoints1"], pair["scores"])
canvas = draw_matches(query_crop, gallery_crop, pair["keypoints0"], pair["keypoints1"], pair["scores"])
order = reorder_head(cosine_order, inlier_counts, k=10)
```

Local weights only (`local_files_only=True`). Download `zju-community/efficientloftr` with
`scripts/download_weights.py efficientloftr`. The checkpoint is gitignored under `weights/` like
other Hub snapshots; it is not part of contest serving.

![EfficientLoFTR correspondences on retrieval pairs](notebooks/eva02/readme_figs/matching_pairs.jpg)

*Green lines are high-score correspondences. An easy true pair (id 1283) and a Rank-1 miss (id 1005)
vs its true positive and vs the distractor. `n_matches` can fire on similar paint; `n_inliers` is
the stricter geometric check.*

![Match evidence vs cosine on the cosine top-10](notebooks/eva02/readme_figs/matching_verifier.jpg)

*Every fold-0 query's cosine top-10, scored by EfficientLoFTR. Same-id pairs have more inliers on
average (56 vs 33), but the clouds overlap. Replacing the embedding Rank-1 with inlier count
**hurts** (Rank-1 0.838 → 0.479). Matching is a visualization / second opinion, not a ranker.
Notebook: `notebooks/eva02/matching.ipynb`.*

### Embedding robustness (post-hoc)

`posthoc/` measures whether the retrieval embedding is stable under input corruptions. The gallery
stays clean. Each query crop is replaced with `k` corrupted copies (severity 1–5). Metrics compare
the corrupted queries to the original query, not to a new trained model.

| Metric | What it measures | Robust if |
| --- | --- | --- |
| `cosine_mean` / `cosine_min` / `cosine_std` | cosine(clean embedding, each of the `k` corrupted embeddings) | close to 1 / 1 / 0 |
| `angular_mean_deg` | mean arccos(cosine) in degrees | close to 0 |
| `pairwise_cosine` | agreement among the `k` copies of one query | close to 1 |
| `overlap` / `jaccard` | set overlap / IoU of clean vs corrupted top-`k` gallery neighbors | close to 1 |
| `kendall` | Kendall-τ on the union of those two top-`k` lists | close to 1 |
| `orig_r1_rank` | rank of the clean Rank-1 gallery image after the query is corrupted | 1 |
| `rank1_image_same` / `rank1_id_same` | same gallery image / same identity still at Rank-1 | 1 |
| `delta_ap` | AP(corrupted query, clean gallery) − AP(clean query) | close to 0 |
| `delta_mean_pos_rank` | shift of true-positive ranks | close to 0 |
| `psnr` / `l1` | pixel distortion of the crop (context for the drop) | — |

Corruptions: `identity`, `gaussian_noise`, `impulse_noise`, `gaussian_blur`, `motion_blur`, `jpeg`,
`brightness`, `contrast`, `saturate`, `downsample`, `occlude`, `rotate`, `crop`, `fog`.

```python
from posthoc import apply_k, compare_query, embed

crops = apply_k(query_rgb, "jpeg", severity=3, k=4, seed=0)
row = compare_query(
    clean_emb, corrupted_emb, gallery_emb, qid=qid, gids=gids, keep=keep, k_neighbors=10
)
```

![Mean cosine between clean and corrupted query embeddings](notebooks/eva02/readme_figs/posthoc_cosine.jpg)

*Gallery stays clean. Each cell is mean cosine(clean query, corrupted query) over fold-0 OOF.
JPEG / blur / color barely move the embedding. Hard crops (`crop` severity 5 → 0.59) and heavy
occlusion / downsample do.*

![Pixel L1 vs embedding cosine under each corruption](notebooks/eva02/readme_figs/posthoc_sensitivity.jpg)

*Same probe: how much the crop has to change in pixels before the 256-D vector moves. Aggressive
re-crops drop cosine far more than an equivalent L1 of noise. Notebook:
`notebooks/eva02/posthoc_stability.ipynb`.*

### Postprocessing

The closed test is a **stream**: the full `test_gallery.csv` is a static database and may be
indexed in advance; each `test_query.csv` row is handled independently. Other test queries and
their results are not available. `submission.csv` is the final ranking after any allowed rerank
(cosine search, then optional pairwise/local-feature or k-reciprocal inside that query's top-K).
It does not have to match a raw nearest-neighbor list from `embeddings.npy`.

Allowed at serve (`postproc.streaming=true`, the default):

- Per-image TTA (flip, rotation, scale, extra bbox context of the current crop).
- Gallery-only mutual-neighbor aggregation (the gallery is static).
- k-reciprocal or GNN rerank of **one query** against the gallery (no query–query graph).

Not used at serve, even though the code still exists for local OOF:

- Average query expansion, including `gallery_only` AQE. Organizers forbid query expansion.
- Joint k-reciprocal / GNN over the whole query batch (`q @ q.T`).
- Clustering or other methods that mix test queries.

`eval.py` raises if AQE is enabled under streaming. OOF notebooks still call `aqe()` / joint
rerankers directly to measure the gap; those numbers are not the submission recipe.
`current_best_tuned` leaves postproc off.

```bash
.venv/bin/python eval.py \
  checkpoint=weights/finetuned/eva02.pt \
  postproc.enabled=true \
  postproc.gallery_aggregation.enabled=true \
  postproc.rerank.kind=k_reciprocal
```

Dense reranking estimates its memory requirement before allocation and stops when it exceeds
`postproc.max_dense_gb`.

### Evaluation artifacts

The evaluation directory contains:

- `embeddings.npy`: contest embedding matrix for `eval.split=test`. Rows are `test_query.csv` in
  file order, then `test_gallery.csv` in file order, with no extra sort. Shape
  `(len(query)+len(gallery), D)`, `float32`. Vectors are L2-normalized after TTA; the official
  scorer also L2-normalizes, so that is optional on their side. `query_id` / `gallery_id` are
  `image_id`. Local `eval.split=val` writes the same file over the held-out fold table instead.
- `retrieval_embeddings.npy`: query and gallery embeddings after enabled expansion stages.
- `distances.npy`: final query-to-gallery distance matrix, when enabled.
- `submission.csv`: wide top-10 table `query_id,gallery_id_1,...,gallery_id_10`. One row per query
  in CSV order, including open-set and refused queries. No `confidence` column. Empty cells and
  skipped queries are invalid here; refusal is not expressed in this file.
- `candidates.csv`: contest accept/refuse file. Columns `query_id,gallery_id,confidence` in that
  order, header required. `confidence` is a monotone similarity score (higher = more confident);
  the range need not be `[0, 1]`. Organizers rank by this value and take the **max-confidence**
  row per `query_id`. Extra rows below that top candidate do not affect F1 or TNR; there is no
  cap on row count. `refusal=` (`none` / `eva02_threshold` / `eva02_model` / `eva02_ensemble`)
  decides which queries are written. A refused query has **no rows**. Empty `gallery_id` and
  placeholder values are not written. Organizers do **not** apply a second threshold to
  `confidence`; including a `query_id` is the accept decision.
- `query.csv` and `gallery.csv`: exact evaluated row order.
- `embedding_order.csv`: sidecar `image_id` list for `embeddings.npy` (not required by the scorer).
- `metrics.json`: run metadata and validation metrics when labels are available.
- `config.yaml`: resolved evaluation configuration.

Validation reports full-gallery mAP (checkpoint selection and Optuna), official mAP@10 over the
submission top-10, mINP, and configured CMC ranks. After junk filtering (same-camera **and**
same-identity gallery images), queries with no remaining valid positive are **excluded from the
mAP@10 / Rank-1 / Rank-5 average**, not scored as AP=0. Gallery images of a *different* identity
on the query camera stay in the ranking. That includes unmarked open-set queries (~20% of the
closed test) and any query whose only gallery positive was junk. Those queries are scored only
through `candidates.csv` (F1, TNR), where the correct output is a refusal (no row). Evaluation
fails if no query has a valid positive.

### Contest inference profile

Organizer timing is the full `extract()` cycle on one vehicle: disk read, decode, bbox crop,
preprocessing, forward, postprocessing, L2. Gallery search and re-ranking are excluded — they scale
with gallery size, not with the embedding model. `profiling/` reproduces that protocol. When
`eval.tta.enabled` is true, the timed `extract()` path averages the same views as `eval.py`:
scales, rotations, flips, and every `eval.tta.context_pcts` crop. The published serving-checkpoint
numbers use TTA off, so they are unchanged.

- `latency_b1`: median of 300 timed batch-1 cycles after 50 warmups, with CUDA synchronize before
  and after every timed sample.
- `throughput`: sustained images/s at batch sizes 1 / 8 / 16 / 32, each run at least 10 seconds.
  The score uses the best FPS.
- Also recorded: peak VRAM, weight-load time (reference), total weight-file bytes, two-run
  determinism.

Performance is 20% of the contest score: `10% × latency_score + 10% × throughput_score`.

| | Full score | Linear | Zero |
| --- | --- | --- | --- |
| Latency (batch=1, full cycle) | ≤ 40 ms | 40–80 ms | > 80 ms |
| Throughput (best FPS) | ≥ 100 | 50–100 | < 50 |

Weight files above 2 GiB are **not admitted** to this performance evaluation (no partial penalty).
The 2 GiB cap is the sum of every inference weight file in the solution directory with suffix
`.pt .pth .bin .onnx .engine .plan .safetensors .ckpt .trt .pb .tflite .npz`. Auxiliary CatBoost
`.cbm` files are reported by `profiling/` but are **not** in that official glob.

The run-time budget is about `latency_b1 × n_test × 3` (~4 minutes at the current closed-test
size). Organizers pass the image directory and CSV (`image_id, x, y, w, h`) as a batch; one
`docker run --gpus all` (or Compose equivalent) must write `submission.csv`, `embeddings.npy`, and
`candidates.csv`. Streaming independence (no query expansion, no use of other test queries) is
checked in source at the final review, not by the launch format.

`notebooks/eva02/inference_profile.ipynb` charts stage costs, latency, FPS vs batch, VRAM, the
weight inventory against the 2 GiB cap, and these performance scores. The contest payload is
`weights/finetuned/eva02.pt` (EMA tensors plus the saved Hydra cfg). A training Lightning `.ckpt`
is not submitted: export it with `scripts/export_serving.py`.

## Open-set refusal

The closed test set is an unmarked open-set probe: about 20% of queries have no corresponding
vehicle in the gallery, and those queries are not flagged in any released file. The remaining 80%
are guaranteed at least one cross-camera positive. Local calibration does **not** copy that 20/80
mixture. It uses the same 5 identity-disjoint OOF folds, nested: models and operating points are
fit with inner CV on four folds, then scored on the held fold. Labels are the 50/50 pairs
(full gallery vs that identity stripped from the gallery). Prevalence on this probe is therefore
balanced, not 20% open.

Contest **F1 and TNR are micro**, not macro: one confusion matrix over the whole query set
(TP/FP/FN/TN summed, then F1/TNR from those totals). There is no per-query or per-identity
average. Both metrics are computed only from `candidates.csv`. `submission.csv` is ignored for
F1/TNR.

Contest F1 is query-level: TP only if the query has a gallery match **and** the highest-confidence
candidate is that identity. Extra candidates below that top row do not change F1 or TNR. Wrong
top-1 on a closed query is FP, not TP. TNR uses only queries with no gallery match. The contest
operating-point score is `0.7 × F1 + 0.3 × TNR`. PR-AUC remains a threshold-free ranking of match vs
no-match queries.

The operating points in `configs/refusal/` maximize the contest score `0.7 × F1 + 0.3 × TNR` on
nested 5-fold inner CV (cosine 0.7011, CatBoost 0.6719). They are the team's accept/refuse rule for
forming `candidates.csv`. Organizers do not re-apply them, and `confidence` is not required to be a
calibrated probability. Outer-OOF metrics concatenate each fold's accept/refuse mask from that
fold's inner-CV threshold; they do not re-apply the mean serving threshold to the evaluated queries.

Outer-OOF on the 50/50 pairs (one confusion matrix over concatenated per-fold decisions):

| Head | contest | F1 | TNR | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Cosine | 0.765 | 0.727 | 0.854 | 0.872 |
| CatBoost | 0.761 | 0.721 | 0.853 | 0.864 |
| Rank-mean | 0.772 | 0.733 | 0.861 | 0.879 |
| Unanimous (3 heads) | 0.755 | 0.702 | 0.879 | 0.763 |

Serving `eva02_ensemble` is the streaming-safe vote of the frozen cosine and CatBoost heads, not the OOF rank-average.

![Outer-OOF refusal head comparison](notebooks/eva02/readme_figs/refusal_bars.jpg)

*Nested 5-fold inner CV, 50/50 match vs stripped-gallery pairs. Contest score is `0.7 × F1 + 0.3 ×
TNR`. “Always accept” has no TNR. Rank-mean is the best OOF calibration; serving `eva02_ensemble`
is the streaming-safe cosine+CatBoost vote, not this rank-average.*

![Outer-OOF precision-recall for match vs no-match](notebooks/eva02/readme_figs/refusal_pr.jpg)

*Threshold-free ranking of “does a gallery match exist?”. Rank-mean PR-AUC 0.879; cosine 0.872;
CatBoost 0.864. Majority vote is a hard decision, not a score, so its curve collapses. Notebook:
`notebooks/eva02/refusal_analysis.ipynb`.*

`refusal/` implements three accept/refuse heads on top of frozen retrieval embeddings, plus
ensembles of those heads. None of them uses `camera_id` or `vehicle_id` as a feature; those labels
exist only while building the training pairs.

1. **Cosine threshold.** Score is the maximum query-gallery cosine (optionally the top-1/top-2
   gap). The operating point maximizes `0.7 × F1 + 0.3 × TNR` on inner-CV scores, never on the
   reported fold.
2. **CatBoost.** A binary classifier on the retrieved set. Training examples are balanced 50/50 by
   scoring the same query against the full gallery (`y=1`) and against the gallery with that
   identity removed (`y=0`). Features are similarity statistics, pairwise/graph descriptors of the
   top-k neighbors, and the concatenated query and top-1 gallery embeddings.
3. **TabM.** A parameter-efficient MLP ensemble after Gorishniy et al., ICLR 2025: shared linear
   weights, BatchEnsemble rank-1 input/output scales, `k` member logits trained jointly, mean
   sigmoid at inference, and train-set feature standardization.
4. **Ensembles.** Rank-average and votes of the three heads are **OOF calibration only** (ranks
   use the query batch). Serving `eva02_ensemble` is a streaming-safe unanimous vote: accept if
   both the frozen cosine and CatBoost heads accept that query.

`eval.py` selects a frozen submit head with `refusal=` (`none`, `eva02_threshold`, `eva02_model`,
`eva02_ensemble`). Serving thresholds and the CatBoost path live in `configs/refusal/`. Default
local `none` writes every query; the Docker image uses `eva02_ensemble`. The mask is applied to
`candidates.csv` only. Each head scores one query against the gallery; serving does not
rank-average across the test query batch.

`notebooks/eva02/refusal_analysis.ipynb` compares the heads and ensembles on EVA02 5-fold OOF.
`notebooks/eva02/inference_profile.ipynb` measures the contest extract() cycle (latency_b1, FPS, VRAM, 2 GiB weights).

## Pretrained weights and offline use

Download supported external weights on a machine with Hugging Face access:

```bash
.venv/bin/python scripts/download_weights.py dinov3_base
.venv/bin/python scripts/download_weights.py dinov3_large
.venv/bin/python scripts/download_weights.py radio
.venv/bin/python scripts/download_weights.py llm2clip
.venv/bin/python scripts/download_weights.py efficientloftr
```

Copy the resulting `weights/` directory to the training machine and set the corresponding
`model.checkpoint_path`. Use `model.local_files_only=true` where supported to prevent network
access. Each download is pinned by file SHA-256 (and a Hub commit for RADIO and LLM2CLIP). A
checksum mismatch after download is an error.

Contest serving weights are not these Hub snapshots. Export EMA tensors from a Lightning training
checkpoint into a compact `.pt`, fit CatBoost on that checkpoint's embeddings (`--full-retrain`) or
on 5-fold OOF packs, then record checksums:

```bash
.venv/bin/python scripts/export_serving.py \
  --checkpoint runs/full/eva02_trial23/checkpoints/last.ckpt \
  --output weights/finetuned/eva02.pt \
  --weights ema \
  --sha256
.venv/bin/python scripts/export_refusal.py \
  --full-retrain \
  --cv runs/cv/eva02_trial23 \
  --checkpoint runs/full/eva02_trial23/checkpoints/last.ckpt \
  --device cuda:1 \
  --output weights/finetuned/eva02_catboost.cbm \
  --sha256
```

`--checkpoint` for the embedding `.pt` defaults to `last_checkpoint` in
`runs/full/eva02_trial23/run_summary.json` when that
file exists, otherwise the path in `runs/cv/eva02_trial23/fold0/val/metrics.json`. The serving `.pt`
is `format=reid-serving`: Hydra `cfg`, ReIDModel `state_dict`, and the exported weight kind.
Optimizer, loops, and the unused raw copy are dropped so the file stays under the
2 GiB contest cap. Without `--full-retrain`, CatBoost is fit on the five fold-OOF embedding packs.
With `--full-retrain`, those same query/gallery CSVs are re-embedded by the serving checkpoint so
the head matches the contest `.pt`. Both use the recipe in
`notebooks/eva02/refusal_analysis.ipynb` (`iterations=200`, `depth=4`, embeddings in the feature
vector). `--sha256` rewrites `weights/finetuned/SHA256SUMS`. `.cbm` is outside the official 2 GiB
suffix glob.

```text
weights/finetuned/
├── SHA256SUMS
├── eva02.pt
└── eva02_catboost.cbm
```

`eva02.pt` is the eval/Docker checkpoint. `eva02_catboost.cbm` is required for
`refusal=eva02_model` and `refusal=eva02_ensemble` (the Docker default). The accept threshold
`0.6719` stays in `configs/refusal/`; it is not retuned at export.

Those binaries are Git LFS objects (see `.gitattributes`). After adding or replacing them without
`--sha256` on the exporter:

```bash
.venv/bin/python scripts/verify_weights.py --root weights/finetuned --write
```

`SHA256SUMS` stays a plain-text file. An empty checksum file is valid only while the directory has
no payload files; the Docker build verifies the tree.

## Contest Docker image

The image may use the network during `docker build`. `docker run` is fully offline:
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and Compose `network_mode: none`. GPU access is
`--gpus all` (Compose `gpus: all`). Python packages are
installed from `requirements/runtime.txt` with `pip install --require-hashes` on public PyPI. That
file is the frozen export of `uv.lock`; regenerate it with `make lock`. The lab `uv.lock` registry
URL is not used at image build time.

The serving `.pt` is copied into the image from `weights/finetuned/`. Do not mount a host
`weights/` directory over `/app/weights`, or the baked files are hidden. Mount only contest data and
the output directory:

```bash
docker compose build
DATA_ROOT=./data OUTPUT_DIR=./runs/submission docker compose run --rm retrieval
```

Equivalent:

```bash
docker run --gpus all --network none \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/runs/submission:/runs/submission" \
  retrieval
```

One command writes `submission.csv`, `embeddings.npy`, and `candidates.csv` from the mounted
`images/` plus query/gallery CSV. The default command is:

```text
checkpoint=/app/weights/finetuned/eva02.pt
data.root=/data
eval.split=test
eval.output_dir=/runs/submission
eval.top_k=10
refusal=eva02_ensemble
```

Override `CHECKPOINT` or pass extra Hydra flags after `retrieval`. Eval loads
`ReIDModel(..., initialize_pretrained=False)`, so Hub/timm pretrained weights are not fetched at
inference. `extra_data/`, tests, notebooks, and training runs are not copied into the image.

## Cross-validation

Run all five `dino_base` folds in parallel. The first argument is the five physical GPU IDs, one
per fold:

```bash
EXPERIMENT=dino_base scripts/train_folds.sh 3,4,5,6,7
```

Each process sees one GPU through `CUDA_VISIBLE_DEVICES` and runs with `trainer.devices=1`. Existing
Hugging Face weights are loaded from `weights/dinov3_base/model.safetensors` with
`model.local_files_only=true`. Override the checkpoint through `MODEL_CHECKPOINT`. Additional Hydra
overrides can be appended after the GPU list:

```bash
RUN_ROOT=runs/cv/dino_base_v1 \
MODEL_CHECKPOINT=/models/dinov3_base/model.safetensors \
scripts/train_folds.sh 3,4,5,6,7 \
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
maximizes the query-weighted OOF retrieval metric. Pass five GPU IDs with `--gpus`; there is no
default assignment:

```bash
.venv/bin/python -m hpo.run_optuna_search \
  --model dinov3_convnext_large \
  --checkpoint weights/dinov3_large/model.safetensors \
  --gpus 3,4,5,6,7 \
  --study-name convnext_large_all \
  --n-trials 200
```

Research of non-streaming methods uses a different study name:

```bash
.venv/bin/python -m hpo.run_optuna_search \
  --model dinov3_convnext_large \
  --checkpoint weights/dinov3_large/model.safetensors \
  --gpus 3,4,5,6,7 \
  --space nonstreaming \
  --study-name convnext_large_nonstreaming \
  --n-trials 200
```

`hpo/optuna_search_space.py` defines two search spaces. `--space contest` is the default and the
only serving-legal search: it forces `postproc.streaming=true`, never enables AQE, and still samples
gallery aggregation plus per-query k-reciprocal/GNN rerank. `--space nonstreaming` is a separate
research mode that forces `postproc.streaming=false` and samples AQE. Contest and nonstreaming
trials must not share a `--study-name`. Both spaces sample input size and crop context, P×K
sampling, backbone optimization controls, pooling, head, every supported loss and its parameters,
optimizer, scheduler, regularization, every augmentation transform and parameter, and TTA. The
selected model architecture, checkpoint, five-fold protocol, data paths, fold assignment, worker
and loader settings, precision, deterministic mode, logging, checkpoint policy, metric protocol,
and output paths stay fixed so trials remain comparable and operational settings do not consume
search trials.

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

![Optuna contest-space search on ConvNeXt-Base](notebooks/eva02/readme_figs/optuna_history.png)

*`convnext_base_all`, contest space, query-weighted 5-fold OOF mAP. Trial 0 is the seeded DINOv3
ConvNeXt-Base run (~0.705). Trial 23 is the best completed trial (0.759). That recipe — not the
ConvNeXt weights — is what `current_best_tuned` applies to EVA02-L-14-336, where OOF mAP becomes
0.847. Later trials did not beat 23. Live view: Optuna Dashboard on the same SQLite file.*

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

Ruff and ty check first-party source, tests, and notebooks. Vendored code under `third_party/` is
the only code exclusion. Function-local imports are forbidden by Ruff rule `PLC0415`. The `dev`
extra includes `matplotlib` and `IPython` so notebook cells type-check and rerun in the same
environment as `make lint`.

## Reproducibility and safety

- Python and core ML dependency versions are pinned exactly in `pyproject.toml`, `uv.lock`, and
  hashed `requirements/runtime.txt`.
- Finetuned serving weights under `weights/finetuned/` are Git LFS objects with `SHA256SUMS`.
- Fold generation is deterministic for a fixed annotation file, seed, and fold count.
- Data fingerprints are persisted and checked when folds or checkpoints are reused.
- Validation identities are disjoint from training identities.
- Partial validation runs are treated as smoke checks and do not report retrieval scores.
- Non-finite embeddings, losses, and reranking results fail explicitly.
- Model adapters validate returned feature count, shape, and configured dimensions.

## Project layout

```text
augmentations/  Config-driven image augmentation pipeline
configs/        Hydra model, loss, optimizer, scheduler, refusal, and experiment presets
dataset/        Annotation validation, folds, datasets, data module, pretrain loaders, and samplers
extra_data/     Optional external identity-labeled crops for pretraining
models/         Backbone adapters, pooling layers, and embedding model
modules/        Lightning module, losses, metrics, inference, optimization, and regularization
interp/         Embedding attribution: cosine Grad-Sim, Grad-Attention rollout, patch occlusion, CAM, Chefer
matching/       EfficientLoFTR pair matching, homography inliers, cosine top-k rerank
notebooks/      EDA plus EVA02 OOF, interp, matching, posthoc, refusal, inference profile; `eva02/readme_figs/` is the README image set
posthoc/        Query-corruption robustness: embedding cosine, neighbor overlap, AP shift
postproc/       Retrieval expansion, aggregation, and reranking
profiling/      Contest extract() timing, weight inventory vs 2 GiB, VRAM, determinism
refusal/        Open-set accept/refuse: cosine threshold, CatBoost, TabM, eval-time mask, contest 0.7 F1 + 0.3 TNR
requirements/   Hashed pip freeze used by the contest Docker image
scripts/        Dataset audit, fold creation, weight download, serving `.pt` / CatBoost export, checksum verification, model checks, zero-shot probes, and CV aggregation
tests/          CPU/offline unit, integration, configuration, and entrypoint tests
third_party/    Vendored upstream implementations
weights/        Local Hub snapshots (gitignored) and `finetuned/` serving artifacts (Git LFS)
Dockerfile      Offline contest serving image
pretrain.py     Hydra pretraining entrypoint on extra data
train.py        Hydra training entrypoint
eval.py         Serving `.pt` / Lightning checkpoint evaluation and retrieval entrypoint
```