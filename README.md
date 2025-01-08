# mlops_project

This is the repository for MLOps. Our project is  based on the Classification task: HOTDOG/ NOT HOTDOG. It's a deep learning task .........


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## How to run the code in docker

1. you should in the `MLOps` directory.

2. Run the below command in your terminal, and you will build the docker image based on my dockerfile.

    ```bash
    docker build -t train_image -f dockerfiles/train.dockerfile .

3. Check your built docker image.
    ```bash
    docker images
    ```
    You will get below result and we can run the train_image now.
    | REPOSITORY  | TAG    | IMAGE ID      | CREATED        | SIZE  |
    |-------------|--------|---------------|----------------|-------|
    | train_image | latest | 311535037766  | 8 minutes ago  | 6.24GB |
4. As we didn't COPY the data into our docker image, we dynamically mount the Host's `data` Directory to the Container's `/data`. 
    ```bash
    '''In Linux system'''
    docker run -v $(pwd)/data:/data -it my_image

    '''In Windows Shell'''
    docker run -v ${PWD}/data:/data -it my_image

5. We can run the docker image right now.
    ```bash
    docker run -v ${PWD}/data:/data -it train_image 