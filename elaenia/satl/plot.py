from pathlib import Path

import numpy as np

from elaenia.plot import plot_matrix


def plot_embeddings(dataset):
    embeddings_dir = Path("data/audio_representations")
    ns = np.load(embeddings_dir / f"evaluation_data_{dataset}_vggish.npz")
    X = ns["X"]
    interval = range(0, len(X), int(len(X) / 250))
    plot_matrix(X[interval])
