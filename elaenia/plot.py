import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(D, sr, y_axis="hz", x_axis="s", **kwargs):
    hop_length = _get_hop_length(_get_n_fft(D))
    D = np.log(D + 1)
    lr.display.specshow(D, x_axis=x_axis, y_axis=y_axis, sr=sr, hop_length=hop_length, **kwargs)


def plot_distance_matrix(D, ax=None):
    dist = distance_matrix(D)
    ax = ax or plt.gca()
    ax.imshow(np.log(dist + 1))


def plot_spectrogram_and_distance_matrix(D, sr):
    "Plot spectrogram and time window distance matrix"
    # TODO: align the subplots

    fig = plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(5, 1, 1)
    plot_spectrogram(D, sr, ax=ax)

    ax = fig.add_subplot(5, 1, (2, 5))
    plot_distance_matrix(D, ax=ax)


def plot_matrix(matrix, title=None):
    plt.figure()
    plt.imshow(np.flipud(matrix.T), interpolation=None)
    plt.colorbar()
    if title != None:
        plt.title(title)
    plt.show()
