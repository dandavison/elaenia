from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from elaenia.recording import Recording
from vendor.tensorflow_models.research.audioset.vggish import mel_features
from vendor.tensorflow_models.research.audioset.vggish import vggish_input


def get_stft_frames(recording: Recording):
    """
    Return log-mel STFT reshaped into frames.

    Note that what I am calling "frames", tensorflow calls "examples".
    What tensorflow calls "frames" are 10ms time windows within each example.

    Output is (n_frames, frame_length, n_bands) = (n_frames, 96, 64)
    """
    return vggish_input.waveform_to_examples(recording.time_series, recording.sampling_rate)


def get_audio_frames(recording: Recording):
    """
    Return raw audio waveform broken into 0.96 s frames.

    Output is (n_frames, frame_length)
    """
    # This is based on the implementation of vggish_input.waveform_to_examples.
    return mel_features.frame(
        recording.time_series,
        window_length=int(round(0.96 * recording.sampling_rate)),
        hop_length=int(round(0.96 * recording.sampling_rate)),
    )


def get_frame_energies(recording):
    """
    Return sum of squared amplitudes within each frame.
    """
    frames = get_audio_frames(recording)
    return (frames ** 2).mean(axis=1)


def get_frame_energy_classes_gmm(recording):
    energies = get_frame_energies(recording).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2)
    gmm = gmm.fit(energies)
    labels = gmm.predict(energies)
    labels = relabel(labels, energies)
    return labels


def get_frame_energy_classes_kmm(recording):
    energies = get_frame_energies(recording).reshape(-1, 1)

    kmm = KMeans(n_clusters=2)
    kmm = kmm.fit(energies)
    labels = kmm.predict(energies)
    labels = relabel(labels, energies)
    return labels


def relabel(labels, values):
    """
    Relabel `labels` such that the sort order of the returned labels is the same as the order of
    the labels when sorted by average `values` value.

    >>> labels = np.array([1, 1, 0, 2])
    >>> values = np.array([1, 2, 4, 3])
    >>> assert relabel(labels, values) == np.array([0, 0, 2, 1])
    """
    assert set(labels) == {0, 1}  # TODO: implement properly
    label_to_mean = {label: values[labels == label].mean() for label in labels}
    if label_to_mean[0] > label_to_mean[1]:
        return 1 - labels
    else:
        return labels
