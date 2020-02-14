from dataclasses import dataclass

import numpy as np

from elaenia.utils import librosa as librosa_utils


@dataclass
class Audio:
    time_series: np.ndarray
    sampling_rate: float

    @classmethod
    def from_file(cls, path):
        ts, sr = librosa_utils.load(path)
        return cls(time_series=ts, sampling_rate=sr)

    def __repr__(self):
        _type = type(self).__name__
        duration = len(self.time_series) / self.sampling_rate
        return f"{_type}(duration={duration}, sampling_rate={self.sampling_rate})"

    def __iter__(self):
        return iter(self.time_series)
