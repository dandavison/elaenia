import numpy as np

from elaenia.dataset import Dataset
from elaenia.pipeline import Transform
from elaenia.vggish import get_audio_frames
from elaenia.vggish import get_stft_frames


class Audio2Windows(Transform):
    def __call__(self, dataset: Dataset) -> Dataset:
        observations = np.array([get_audio_frames(audio) for audio in dataset.observations])
        return Dataset(observations=observations, labels=dataset.labels)


class Audio2Spectrogram(Transform):
    def __call__(self, dataset: Dataset) -> Dataset:
        observations = np.array([get_stft_frames(audio) for audio in dataset.observations])
        return Dataset(observations=observations, labels=dataset.labels)


class Audio2VGGishEmbeddings(Transform):
    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
