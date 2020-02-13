import numpy as np

from elaenia.dataset import Dataset
from elaenia.pipeline import Transformer
from elaenia.vggish import get_audio_frames
from elaenia.vggish import get_stft_frames


class Audio2Windows(Transformer):
    def transform(self, dataset: Dataset) -> Dataset:
        observations = np.array([get_audio_frames(audio) for audio in dataset.observations])
        return Dataset(observations=observations, labels=dataset.labels)


class Audio2Spectrogram(Transformer):
    def transform(self, dataset: Dataset) -> Dataset:
        observations = np.array([get_stft_frames(audio) for audio in dataset.observations])
        return Dataset(observations=observations, labels=dataset.labels)


class Audio2VGGishEmbeddings(Transformer):
    def transform(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
