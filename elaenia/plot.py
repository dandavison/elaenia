import librosa as lr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def stft(y, n_fft):
    return np.abs(
        lr.stft(
            y, n_fft=n_fft, hop_length=_get_hop_length(n_fft), win_length=_get_win_length(n_fft)
        )
    )


def _get_n_fft(D):
    half_n_fft = D.shape[0] - 1
    assert _is_power_of_2(half_n_fft)
    return half_n_fft << 1


def _get_hop_length(n_fft):
    assert _is_power_of_2(n_fft)
    return n_fft >> 2


def _get_win_length(n_fft):
    assert _is_power_of_2(n_fft)
    return n_fft


def _is_power_of_2(n):
    return n and not (n & (n - 1))


def dist(s1, s2):
    "Distance between spectra"
    return ((s1 - s2) ** 2).sum()


def distance_matrix(D):
    "Matrix of distances between spectra at different time windows"
    n_freq_bins, n_time_bins = D.shape
    d = np.zeros((n_time_bins, n_time_bins))
    for i in range(n_time_bins):
        for j in range(i + 1):
            d[i, j] = d[j, i] = np.linalg.norm(D[:, i] - D[:, j])
    return 1 - d / d.max()


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
