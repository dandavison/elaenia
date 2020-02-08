import matplotlib.pyplot as plt
import numpy as np

import elaenia.stft
import elaenia.plot
from elaenia.satl.dataset import DatasetRecording


def plot_spectrogram_and_embeddings_and_classifications(recording: DatasetRecording):
    fig, axes = plt.subplots(nrows=3)
    spect_ax = axes[2]
    embed_ax = axes[1]
    preds_ax = axes[0]

    stft = elaenia.stft.stft(recording.time_series, n_fft=1024)

    plot_spectrogram(recording, ax=spect_ax, stft=stft)
    plot_embeddings(recording, ax=embed_ax, stft=stft)


def plot_spectrogram(recording, ax, stft):
    n_freq_frames, n_time_frames = stft.shape
    # Crop high frequencies
    # n_freq_frames is assumed to be power of 2 plus 1
    # n_freq_frames = ((n_freq_frames - 1) >> 1) + 1
    return elaenia.plot.plot_spectrogram(stft[:n_freq_frames, :], sr=recording.sampling_rate, ax=ax)


def plot_embeddings(recording, ax, stft):
    embeddings = recording.frame_vggish_embeddings.T
    assert embeddings.shape[0] == 128
    embeddings = align_vggish_frames_to_stft_time_frames(embeddings)
    return imshow(ax, embeddings)


def align_vggish_frames_to_stft_time_frames(vggish_frames, stft):
    """
    Approximately align the VGGish frames (0.96 s) with the STFT time frames (shorter than that).
    The input is a (d, n_frames) matrix, where d is the dimension of whatever data is associated
    with each frame. For example, if the input is classification labels, then d will be 1.
    Alternatively, if the input is VGGish frame embeddings, then d will be 128 (since that is the
    size of the embedding layer in Google's VGGish network).
    """
    d, vggish_n_frames = embeddings.shape
    stft_n_freq_frames, stft_n_time_frames = stft.shape
    stft_time_frames_per_vggish_frame = int(stft_n_time_frames / vggish_n_frames)
    return vggish_frames.repeat(stft_time_frames_per_vggish_frame, axis=1)


def imshow(ax, matrix):
    return ax.imshow(np.flipud(matrix), interpolation=None, aspect="auto")
