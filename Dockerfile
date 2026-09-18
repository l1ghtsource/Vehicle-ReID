FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements/runtime.txt /app/requirements/runtime.txt
RUN python -m pip install --no-cache-dir --require-hashes -r requirements/runtime.txt

COPY pyproject.toml /app/pyproject.toml
COPY models /app/models
COPY modules /app/modules
COPY dataset /app/dataset
COPY augmentations /app/augmentations
COPY postproc /app/postproc
COPY refusal /app/refusal
COPY third_party /app/third_party
COPY scripts /app/scripts
COPY configs /app/configs
COPY train.py pretrain.py eval.py /app/
COPY weights/finetuned /app/weights/finetuned

RUN python -m pip install --no-cache-dir --no-deps . \
    && python scripts/verify_weights.py --root /app/weights/finetuned

ENTRYPOINT ["python", "eval.py"]
CMD ["checkpoint=/app/weights/finetuned/model.ckpt", "data.root=/data", "eval.split=test", "eval.output_dir=/runs/submission"]
