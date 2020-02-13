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

    @property
    def n(self) -> int:
        return len(self.observations)

    def compute_accuracy(self, predictions) -> float:
        return np.mean(p == t for p, t in zip(predictions, self.labels))

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

    @staticmethod
    def unpack_objects(array: np.ndarray) -> np.ndarray:
        """
        If `array` contains objects, covert them to arrays, adding one
        dimension, so that the result is a 2D numerical numpy array.
        The objects must be iterable.
        """
        if array.dtype == np.object:
            return np.array([np.array(obj) for obj in array])
        else:
            return array
