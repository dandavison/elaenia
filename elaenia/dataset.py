import numpy as np


class Dataset:
    """
    A dataset comprises the union of its training and testing subsets, which are themselves
    datasets. So, e.g.
    dataset.n == len(dataset.observations) == len(dataset.labels)
    dataset.n == dataset.training_dataset.n + dataset.testing_dataset.n
    """

    def __init__(self, observations: np.ndarray, labels: np.ndarray, training_proportion=0.8):
        self.observations = observations
        self.labels = labels
        self.training_proportion = training_proportion
        self._validate()

    def _validate(self):
        assert len(self.observations) == len(self.labels)
        if self.observations.dtype == np.object:
            assert len(self.observations.shape) == 1
        else:
            assert len(self.observations.shape) > 1
        assert len(self.labels.shape) == 1

    def __repr__(self):
        _type = type(self).__name__
        return f"{_type}(n={self.n}, labels=[{repr(self.labels[0])}, ...])"

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def _training_rows(self) -> np.ndarray:
        n_training = int(self.training_proportion * self.n)
        return np.array([i < n_training for i in range(self.n)])

    @property
    def training_dataset(self) -> "Dataset":
        is_training = self._training_rows
        return Dataset(
            observations=self.observations[is_training], labels=self.labels[is_training]
        )

    @property
    def testing_dataset(self) -> "Dataset":
        is_testing = ~self._training_rows
        return Dataset(observations=self.observations[is_testing], labels=self.labels[is_testing])

    # Hack: This method is only here to make the type checker pass
    # under composition of Transforms and Learners.
    def predict(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError
