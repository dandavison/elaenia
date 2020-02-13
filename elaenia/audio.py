from dataclasses import dataclass

import numpy as np


@dataclass
class Audio:
    time_series: np.ndarray
    sampling_rate: float

    def __repr__(self):
        _type = type(self).__name__
        return f"{_type}(length={len(self.time_series)}, sampling_rate={self.sampling_rate})"

    def __iter__(self):
        return iter(self.time_series)
