FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml /app/
COPY models /app/models
COPY modules /app/modules
COPY dataset /app/dataset
COPY augmentations /app/augmentations
COPY postproc /app/postproc
COPY third_party /app/third_party
RUN pip install --no-cache-dir .
COPY configs /app/configs
COPY scripts /app/scripts
COPY train.py eval.py /app/
ENV NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "eval.py"]
