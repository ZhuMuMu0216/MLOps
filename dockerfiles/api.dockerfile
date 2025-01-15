FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc bash && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /MLOps/

COPY models models
COPY src src
COPY keys keys
COPY requirements_api.txt requirements_api.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN pip install -r requirements_api.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.mlops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
