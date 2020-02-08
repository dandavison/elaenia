import matplotlib.pyplot as plt
import numpy as np

import elaenia.stft
import elaenia.plot
from elaenia.satl.experiment_recording import ExperimentRecording


def plot_spectrogram_and_embeddings_and_classifications(recording: ExperimentRecording):
    fig, axes = plt.subplots(nrows=4)
    spect_ax = axes[3]
    embed_ax = axes[2]
    preds_ax = axes[1]
    truth_ax = axes[0]

    stft = elaenia.stft.stft(recording.time_series, n_fft=1024)

    plot_spectrogram(recording, ax=spect_ax, stft=stft)
    plot_embeddings(recording, ax=embed_ax, stft=stft)
    plot_predictions(recording, ax=preds_ax, stft=stft)
    plot_truth(recording, ax=truth_ax, stft=stft)
    truth_ax.set_title(recording.id)
    for ax in axes:
        ax.label_outer()


def plot_spectrogram(recording, ax, stft):
    n_freq_frames, n_time_frames = stft.shape
    # Crop high frequencies
    # n_freq_frames is assumed to be power of 2 plus 1
    # n_freq_frames = ((n_freq_frames - 1) >> 1) + 1
    return elaenia.plot.plot_spectrogram(stft[:n_freq_frames, :], sr=recording.sampling_rate, ax=ax)


def plot_embeddings(recording, ax, stft):
    embeddings = recording.frame_vggish_embeddings.T
    assert embeddings.shape[0] == 128
    embeddings = align_frames_to_stft_time_frames(embeddings, stft)
    return imshow(ax, embeddings)


def plot_predictions(recording, ax, stft):
    preds = recording.frames_predicted_integer_labels[np.newaxis, :]
    preds = align_frames_to_stft_time_frames(preds, stft)
    return imshow_integer_labels(ax, preds, recording)


def plot_truth(recording, ax, stft):
    truth = np.array([recording.predicted_integer_label, recording.integer_label])[np.newaxis, :]
    truth = align_frames_to_stft_time_frames(truth, stft)
    return imshow_integer_labels(ax, truth, recording)


def align_frames_to_stft_time_frames(frames, stft):
    """
    Approximately align frames with the STFT time frames. The input is a (d, n_frames) matrix,
    where d is the dimension of whatever data is associated with each frame. For example, if the
    input is classification labels, then d will be 1. Alternatively, if the input is VGGish frame
    embeddings, then d will be 128 (since that is the size of the embedding layer in Google's
    VGGish network). In both bhose cases, the frame length is 0.96 s, which will typically be
    longer than the STFT frame length.
    """
    d, n_frames = frames.shape
    stft_n_freq_frames, stft_n_time_frames = stft.shape
    assert stft_n_time_frames > n_frames
    stft_time_frames_per_frame = int(stft_n_time_frames / n_frames)
    return frames.repeat(stft_time_frames_per_frame, axis=1)


def imshow(ax, matrix, **kwargs):
    return ax.imshow(np.flipud(matrix), interpolation=None, aspect="auto", **kwargs)


def imshow_integer_labels(ax, matrix, recording):
    integer_label_set = (recording.dataset.experiment.train_set.integer_label_set |
                         recording.dataset.experiment.test_set.integer_label_set)
    vmin, vmax = min(integer_label_set), max(integer_label_set)
    return imshow(ax, matrix, vmin=vmin, vmax=vmax)
