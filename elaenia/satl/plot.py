import matplotlib.pyplot as plt
import numpy as np

import elaenia.stft
import elaenia.plot
from elaenia.satl.dataset import DatasetRecording


def plot_spectrogram_and_embeddings_and_classifications(recording: DatasetRecording):
    fig, axes = plt.subplots(nrows=2)
    embed_ax = axes[0]
    spect_ax = axes[1]

    stft = elaenia.stft.stft(recording.time_series, n_fft=1024)
    plot_spectrogram(recording, stft=stft, ax=spect_ax)
    plot_embeddings(recording, stft=stft, ax=embed_ax)


def plot_spectrogram(recording, stft, ax):
    n_freq_frames, n_time_frames = stft.shape
    # Crop high frequencies
    # n_freq_frames is assumed to be power of 2 plus 1
    # n_freq_frames = ((n_freq_frames - 1) >> 1) + 1
    return elaenia.plot.plot_spectrogram(stft[:n_freq_frames, :], sr=recording.sampling_rate, ax=ax)


def plot_embeddings(recording, stft, ax):
    embeddings = recording.frame_vggish_embeddings.T

    # Approximately align the VGGish frames (0.96 s) with the STFT time frames (shorter than that).
    vggish_embedding_dimension, vggish_n_frames = embeddings.shape
    assert vggish_embedding_dimension == 128
    stft_n_freq_frames, stft_n_time_frames = stft.shape
    stft_time_frames_per_vggish_frame = int(stft_n_time_frames / vggish_n_frames)
    embeddings = embeddings.repeat(stft_time_frames_per_vggish_frame, axis=1)

    img = imshow(ax, embeddings)


def imshow(ax, matrix):
    return ax.imshow(np.flipud(matrix), interpolation=None, aspect="auto")
