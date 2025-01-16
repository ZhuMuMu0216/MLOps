# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

COPY models/ /models/
COPY src/ /src/
# COPY keys/ /keys/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose
ENV WANDB_API_KEY=edac1fed7ead25ecc22f33edb1468b626a2168e5

# pull data
RUN dvc pull -v --force

ENTRYPOINT ["python", "-u", "src/mlops_project/train.py", "entrypoint"]
