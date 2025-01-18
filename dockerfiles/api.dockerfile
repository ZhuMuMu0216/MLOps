FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc bash && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 1. 安装系统构建依赖 + curl（用于安装 Rust）
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential gcc bash curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. 安装 Rust (通过 rustup)。也可以用 apt-get install cargo rustc (版本可能较旧)
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    # 为了后续命令行能找到 cargo，把它加入 PATH
    echo 'source $HOME/.cargo/env' >> /root/.bashrc


# 这里显式将 cargo bin 路径写入环境变量
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
