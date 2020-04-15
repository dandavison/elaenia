import subprocess
from pathlib import Path

PROJECT = "elaenia"
ENVIRONMENT = "elaenia-environment"
LOCATION = "us-central1"


def import_dag(path: Path):
    _composer__environments__storage__dags("import", "--source", path)


def list_dags():
    _composer__environments__storage__dags("list")


def delete_dag(dag: str):
    _composer__environments__storage__dags("delete", dag)


def list_tasks(dag: str):
    _composer__environments__run("list_tasks", dag)


###############################################################################


# gcloud composer environments run (ENVIRONMENT : --location=LOCATION)
#         SUBCOMMAND [GCLOUD_WIDE_FLAG ...] [-- CMD_ARGS ...]
#
# E.g.
#
# gcloud composer environments run elaenia-environment --location us-central1 \
#    list_tasks --project=elaenia -- composer_sample_dag


def _composer__environments__run(subcommand, *args):
    _gcloud(
        "composer",
        "environments",
        "run",
        ENVIRONMENT,
        "--location",
        LOCATION,
        subcommand,
        "--project",
        PROJECT,
        "--",
        *args,
    )


# gcloud composer environments storage dags COMMAND [GCLOUD_WIDE_FLAG ...]

# gcloud composer environments storage dags list
#     (--environment=ENVIRONMENT : --location=LOCATION)
#     [GCLOUD_WIDE_FLAG ...]

# gcloud composer environments storage dags import --source=SOURCE
#     (--environment=ENVIRONMENT : --location=LOCATION)
#     [--destination=DESTINATION] [GCLOUD_WIDE_FLAG ...]

# gcloud composer environments storage dags delete [TARGET]
#     (--environment=ENVIRONMENT : --location=LOCATION)
#     [GCLOUD_WIDE_FLAG ...]


def _composer__environments__storage__dags(subcommand, *args):
    _gcloud(
        "composer",
        "environments",
        "storage",
        "dags",
        subcommand,
        "--environment",
        ENVIRONMENT,
        "--location",
        LOCATION,
        *args,
    )


def _gcloud(*args):
    cmd = ["gcloud", *args]
    print(" ".join(cmd))
    subprocess.run(cmd)
