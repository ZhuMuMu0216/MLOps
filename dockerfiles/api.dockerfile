FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ /src/
COPY requirements_api.txt requirements_api.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN pip install -r requirements_api.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE 8080
ENTRYPOINT ["uvicorn", "src.mlops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
