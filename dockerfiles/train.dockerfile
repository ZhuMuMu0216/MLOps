# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git bash && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    gcc

# 1. Install system build dependencies + curl (used to install Rust)
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential gcc bash curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Rust (via rustup). Alternatively, use apt-get install cargo rustc (but the version may be outdated)
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    # Add cargo to PATH for subsequent commands to locate it
    echo 'source $HOME/.cargo/env' >> /root/.bashrc

# Explicitly add the cargo bin directory to the PATH environment variable
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /MLOps/

# code related
COPY models models
COPY src src
COPY keys keys
COPY data data
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY configs configs

RUN set -e && \
    pip install -r requirements.txt --no-cache-dir --verbose && \
    pip install . --no-deps --no-cache-dir --verbose && \
    rm -rf build dist src/mlops_project.egg-info

ENV WANDB_API_KEY=edac1fed7ead25ecc22f33edb1468b626a2168e5


ENTRYPOINT ["python", "-u", "src/mlops_project/train.py", "entrypoint"]
