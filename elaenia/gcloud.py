import subprocess
from itertools import chain
from pathlib import Path


def import_dag(path: Path):
    _dags("import", "--source", path)


def list_dags():
    _dags("list")


def delete_dag(dag: str):
    _dags("delete", dag)


def _dags(*args):
    _gcloud("composer", "environments", "storage", "dags", *args)


def _gcloud(*args):
    options = {
        "--project": "elaenia",
        "--environment": "elaenia-environment",
        "--location": "us-central1",
    }
    cmd = ["gcloud", *args, *chain(*options.items())]
    print(" ".join(cmd))
    subprocess.run(cmd)
