from dataclasses import dataclass

import numpy as np


@dataclass
class Audio:
    time_series: np.ndarray
    sampling_rate: float

    def __repr__(self):
        _type = type(self).__name__
        duration = len(self.time_series) / self.sampling_rate
        return f"{_type}(duration={duration}, sampling_rate={self.sampling_rate})"

    def __iter__(self):
        return iter(self.time_series)
