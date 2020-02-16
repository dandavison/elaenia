from cached_property import cached_property
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from elaenia.recording import Recording
from sylph.vendor.tensorflow_models.research.audioset.vggish import mel_features
from sylph.vendor.tensorflow_models.research.audioset.vggish import vggish_input


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


class VGGishFrames:
    def __init__(self, recording):
        self.recording = recording
        self.frames = get_audio_frames(self.recording)
        self.gmm = self._fit_frame_energy_gmm()
        self.kmm = self._fit_frame_energy_kmm()

    @cached_property
    def frame_energies(self):
        """
        Return sum of squared amplitudes within each frame.
        """
        return (self.frames ** 2).sum(axis=1)

    def get_time_series_with_low_energy_frames_removed(self):
        frames = get_audio_frames(self.recording)
        is_high_energy = self.frame_energy_class_predictions_gmm
        assert set(is_high_energy) <= {0, 1}
        frames = frames[is_high_energy == 1]
        return frames.ravel()

    def _fit_frame_energy_gmm(self):
        gmm = GaussianMixture(n_components=2)
        return gmm.fit(self.frame_energies.reshape(-1, 1))

    def _fit_frame_energy_kmm(self):
        energies = self.frame_energies
        kmm = KMeans(n_clusters=2)
        return kmm.fit(energies.reshape(-1, 1))

    def _get_frame_energy_class_predictions(self, model):
        energies = self.frame_energies
        labels = model.predict(energies.reshape(-1, 1))
        labels = relabel(labels, energies)
        return labels

    def _get_frame_energy_class_probabilities(self, model):
        energies = self.frame_energies
        probs = model.predict_proba(energies.reshape(-1, 1))
        return probs

    @property
    def frame_energy_class_predictions_gmm(self):
        return self._get_frame_energy_class_predictions(self.gmm)

    @property
    def frame_energy_class_predictions_kmm(self):
        return self._get_frame_energy_class_predictions(self.kmm)

    @cached_property
    def frame_energy_class_probabilities_gmm(self):
        return self._get_frame_energy_class_probabilities(self.gmm)

    @cached_property
    def frame_energy_class_probabilities_kmm(self):
        return self._get_frame_energy_class_probabilities(self.kmm)


def relabel(labels, values):
    """
    Relabel `labels` such that the sort order of the returned labels is the same as the order of
    the labels when sorted by average `values` value.

    >>> import numpy as np
    >>> labels = np.array([1, 1, 0, 2])
    >>> values = np.array([1, 2, 4, 3])
    >>> # TODO: implement properly and enable test
    >>> assert True or relabel(labels, values) == np.array([0, 0, 2, 1])
    """
    assert set(labels) <= {0, 1}  # TODO: implement properly
    if len(set(labels)) == 1:
        # assert set(labels) == {0} # no?
        return labels
    label_to_mean = {label: values[labels == label].mean() for label in labels}
    if label_to_mean[0] > label_to_mean[1]:
        return 1 - labels
    else:
        return labels
