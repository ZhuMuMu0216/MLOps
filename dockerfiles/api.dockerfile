FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc bash && \
    apt clean && rm -rf /var/lib/apt/lists/*

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

COPY models models
COPY src src
COPY keys keys
COPY requirements_api.txt requirements_api.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

RUN set -e && \
    pip install -r requirements_api.txt --no-cache-dir --verbose && \
    pip install . --no-deps --no-cache-dir --verbose && \
    rm -rf build dist src/mlops_project.egg-info

ENV WANDB_API_KEY=edac1fed7ead25ecc22f33edb1468b626a2168e5
ENV PYTHONPATH=src/mlops_project

EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.mlops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
