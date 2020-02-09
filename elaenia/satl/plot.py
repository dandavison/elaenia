from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import elaenia.stft
import elaenia.plot
from elaenia.satl.experiment import Experiment
from elaenia.satl.experiment_recording import ExperimentRecording


def plot_experiment(experiment: Experiment, dir: str):
    dir = Path(dir)
    plt.ioff()
    for dataset in [experiment.train_set, experiment.test_set]:
        plot_dataset(dataset, dir / dataset.name)


def plot_dataset(dataset, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    for recording in dataset.recordings:
        fig_path = dir / recording.id.replace("/", "_").replace(".mp3", ".png")
        if fig_path.exists():
            continue
        print(dataset.name, recording.id)
        plot_spectrogram_and_embeddings_and_classifications(recording)
        plt.savefig(fig_path)
        plt.close()


def plot_spectrogram_and_embeddings_and_classifications(recording: ExperimentRecording):
    fig, axes = plt.subplots(nrows=4)
    truth_ax = axes[0]
    preds_ax = axes[1]
    embed_ax = axes[2]
    spect_ax = axes[3]

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
    return elaenia.plot.plot_spectrogram(
        stft[:n_freq_frames, :], sr=recording.sampling_rate, ax=ax
    )


def plot_embeddings(recording, ax, stft):
    embeddings = recording.frame_vggish_embeddings.T
    assert embeddings.shape[0] == 128
    return imshow(embeddings, ax)


def plot_predictions(recording, ax, stft):
    preds = recording.frames_predicted_integer_labels[np.newaxis, :]
    return imshow_integer_labels(preds, ax, recording)


def plot_truth(recording, ax, stft):
    truth = np.array([recording.predicted_integer_label, recording.integer_label])[np.newaxis, :]
    return imshow_integer_labels(truth, ax, recording)


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


def imshow(matrix, ax, stft, **kwargs):
    matrix = align_frames_to_stft_time_frames(matrix, stft)
    return ax.imshow(np.flipud(matrix), interpolation=None, aspect="auto", **kwargs)


def imshow_integer_labels(matrix, ax, recording):
    integer_label_set = (
        recording.dataset.experiment.train_set.integer_label_set
        | recording.dataset.experiment.test_set.integer_label_set
    )
    vmin, vmax = min(integer_label_set), max(integer_label_set)
    return imshow(matrix, ax, stft, vmin=vmin, vmax=vmax)
