import librosa as lr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


N_FFT = 2 ** 10


def stft(y):
    ss = np.abs(lr.stft(y, n_fft=N_FFT))
    ss = np.log(ss + 1)
    return ss


def dist(s1, s2):
    "Distance between spectra"
    return ((s1 - s2) ** 2).sum()


def distance_matrix(ss):
    "Matrix of distances between spectra at different time windows"
    n_freq_bins, n_time_bins = ss.shape
    d = np.zeros((n_time_bins, n_time_bins))
    for i in range(n_time_bins):
        for j in range(i + 1):
            d[i, j] = d[j, i] = np.linalg.norm(ss[:, i] - ss[:, j])
    return 1 - d / d.max()


def plot_spectrogram(ss, y_axis="hz", x_axis="s", **kwargs):
    lr.display.specshow(ss, x_axis=x_axis, y_axis=y_axis, **kwargs)


def plot_distance_matrix(ss, ax=None):
    d = distance_matrix(ss)
    ax = ax or plt.gca()
    ax.imshow(np.log(d + 1))


def plot_spectrogram_and_distance_matrix(ss):
    "Plot spectrogram and time window distance matrix"
    # TODO: align the subplots

    fig = plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(5, 1, 1)
    plot_spectrogram(ss, ax=ax)

    ax = fig.add_subplot(5, 1, (2, 5))
    plot_distance_matrix(ss, ax=ax)
