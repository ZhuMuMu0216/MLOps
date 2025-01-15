#!/bin/bash
dvc pull --force # download data
python -u ./src/mlops_project/train.py entrypoint