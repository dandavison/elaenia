import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from more_itertools import run_length
import numpy as np
from toolz import groupby

import elaenia.utils.dask
from elaenia.pipelines.birdnet import make_birdnet_embeddings_training_pipeline
from sylph.xeno_quero.dataset import XenoQueroDataset


def plot_probabilities(output, line_color="red"):
    dataset = output["transformed_testing_dataset"]
    dataset["prediction_probs"] = output["transformed_testing_dataset_prediction_probs"]
    row_species, row_species_counts = zip(*run_length.encode(dataset.labels))
    idx = [sorted(row_species).index(sp) for sp in row_species]
    plt.imshow(dataset["prediction_probs"][:, idx], aspect="auto")
    row_species_offsets = [0] + list(np.cumsum(row_species_counts))
    for col in range(len(row_species)):
        xlim = [col - 0.5, col + 0.5]
        ylim = [row_species_offsets[col], row_species_offsets[col + 1]]
        plt.hlines(ylim, xmin=xlim[0], xmax=xlim[1], colors=line_color)
        plt.vlines(xlim, ymin=ylim[0], ymax=ylim[1], colors=line_color)


if __name__ == "__main__":
    elaenia.utils.dask.activate_fargate_cluster()
    dataset = XenoQueroDataset.from_species_globs(["Xiphorhynchus-*"], training_proportion=0.8)
    pipeline = make_birdnet_embeddings_training_pipeline()
    output = pipeline.run(dataset)
    metrics = pipeline.get_metrics(dataset, output)
    with Path("BirdNet-TL-Xiphorhynchus.pickle").open("wb") as fp:
        pickle.dump({"output": output, "metrics": metrics}, fp)
    print(metrics)
