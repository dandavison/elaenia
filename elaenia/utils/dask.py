from dask.distributed import Client
from dask_cloudprovider import FargateCluster

import elaenia.utils
import sylph.utils


def activate_fargate_cluster():
    cluster = FargateCluster(image="dandavison7/elaenia", n_workers=1)
    client = Client(cluster)
    print(cluster)
    print(cluster.dashboard_link)
    return client


def provision_workers(client):
    egg_paths = [elaenia.utils.get_egg_path(), sylph.utils.get_egg_path()]
    for egg_path in egg_paths:
        client.upload_file(egg_path)
