from typing import Optional

import numpy as np


class Dataset:
    """
    A dataset comprises the union of its training and testing subsets, which are themselves
    datasets. So, e.g.
    dataset.n == len(dataset.observations) == len(dataset.labels)
    dataset.n == dataset.training_dataset.n + dataset.testing_dataset.n
    """

    def __init__(
        self,
        observations: np.ndarray,
        labels: np.ndarray,
        training_rows: Optional[np.ndarray] = None,
        training_proportion: Optional[float] = None,
    ):
        self.observations = observations
        self.labels = labels
        self._training_rows = training_rows
        self.training_proportion = training_proportion
        self._validate()

    def __repr__(self):
        _type = type(self).__name__
        return f"{_type}(n={self.n}, labels=[{repr(self.labels[0])}, ...])"

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def training_rows(self) -> np.ndarray:
        assert self._training_rows is not None or self.training_proportion is not None
        if self._training_rows is not None:
            return self._training_rows
        else:
            n_training = int(self.training_proportion * self.n)
            return np.array([i < n_training for i in range(self.n)])

    @property
    def training_dataset(self) -> "Dataset":
        is_training = self.training_rows
        return Dataset(
            observations=self.observations[is_training], labels=self.labels[is_training]
        )

    @property
    def testing_dataset(self) -> "Dataset":
        is_testing = ~self.training_rows
        return Dataset(observations=self.observations[is_testing], labels=self.labels[is_testing])

    def _validate(self):
        assert isinstance(self.observations, np.ndarray)
        assert isinstance(self.labels, np.ndarray)
        assert len(self.observations) == len(self.labels)
        if self.observations.dtype == np.object:
            assert len(self.observations.shape) == 1
        else:
            assert len(self.observations.shape) > 1
        assert len(self.labels.shape) == 1
