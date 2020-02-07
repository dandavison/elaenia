import matplotlib.pyplot as plt
import numpy as np

import elaenia.stft
from elaenia.plot import plot_matrix
from elaenia.satl.dataset import DatasetRecording


def plot_spectrogram_and_embeddings_and_classifications(recording: DatasetRecording):
    fig, axes = plt.subplots(nrows=2)
    embed_ax = axes[0]
    spect_ax = axes[1]
    stft = elaenia.stft.stft(recording.time_series, n_fft=1024)
    plot_spectrogram(stft, ax=spect_ax)
    plot_embeddings(recording, ax=embed_ax, stft=stft)


def plot_spectrogram(stft, ax):
    imshow(ax, np.log(stft + 1))


def plot_embeddings(recording, ax, stft):
    embeddings = recording.frame_vggish_embeddings.T

    print(f"stft: {stft.shape}")
    print(f"embeddings: {embeddings.shape}")

    # align the embedding with the spectrogram.
    vggish_embedding_dimension, vggish_n_frames = embeddings.shape
    assert vggish_embedding_dimension == 128
    stft_n_freq_frames, stft_n_time_frames = stft.shape
    stft_time_frames_per_vggish_frame = int(stft_n_time_frames / vggish_n_frames)
    embeddings = embeddings.repeat(stft_time_frames_per_vggish_frame, axis=1)

    print(f"stft_time_frames_per_vggish_frame: {stft_time_frames_per_vggish_frame}")
    print(f"embeddings.shape: {embeddings.shape}")
    print(f"stft.shape: {stft.shape}")

    img = imshow(ax, embeddings)


def imshow(ax, matrix):
    return ax.imshow(np.flipud(matrix), interpolation=None, aspect="auto")
