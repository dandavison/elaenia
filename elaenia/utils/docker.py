from pathlib import Path

import docker

from sylph.utils.color import red
from elaenia.utils import getenv


def get_docker_client():
    client = docker.from_env()
    client.login(
        username=getenv("DOCKER_REGISTRY_USERNAME"),
        password=getenv("DOCKER_REGISTRY_PASSWORD"),
        registry=getenv("DOCKER_REGISTRY_URL", "https://index.docker.io/v1/"),
    )
    return client


def get_dockerfile():
    path = Path(__file__).resolve().parents[2] / "Dockerfile"
    assert path.exists()
    return path


def build_docker_image():
    client = get_docker_client()
    red("Building docker image...")
    return client.images.build(path=str(get_dockerfile().parent), tag="elaenia")
