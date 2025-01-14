import os
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.10.16"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    conda_command = "conda.bat" if WINDOWS else "conda"  # Windows 使用 conda.bat
    ctx.run(
        f"{conda_command} create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    python_cmd = "python" if WINDOWS else "python3"  # Windows 默认使用 python
    ctx.run(f"{python_cmd} -m pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run(f"{python_cmd} -m pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run(f"{python_cmd} -m pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    python_cmd = "python" if WINDOWS else "python3"
    ctx.run(f'{python_cmd} -m pip install -e .["dev"]', echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, epoch: int = 10) -> None:
    """Train model."""
    python_cmd = "python" if WINDOWS else "python3"
    base_command = f"{python_cmd} src/{PROJECT_NAME}/train.py entrypoint --epoch {epoch}"
    
    ctx.run(base_command, echo=True, pty=not WINDOWS)
# Use command
# invoke train --epochs '2'



@task
def test(ctx: Context) -> None:
    """Run tests."""
    python_cmd = "python" if WINDOWS else "python3"
    ctx.run(f"{python_cmd} -m coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run(f"{python_cmd} -m coverage report -m", echo=True, pty=not WINDOWS)

@task
def git(ctx, message):
    '''git add, commit and push'''
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    docker_command = "docker.exe" if WINDOWS else "docker"  # Windows 环境中可能需要 docker.exe
    ctx.run(
        f"{docker_command} build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"{docker_command} build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )

@task
def pull_data(ctx):
    '''Pull data from remote storage'''
    ctx.run("dvc pull")

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)



@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
