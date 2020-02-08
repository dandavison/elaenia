from functools import cached_property

import numpy as np


class Results:
    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset

    @property
    def recordings_predicted_integer_labels(self):
        return self._data["y_pred"]

    @property
    def recordings_true_integer_labels(self):
        return self._data["y_true"]

    @property
    def frames_predicted_integer_labels(self):
        return self._data["pred"]

    @cached_property
    def _data(self):
        return np.load(self.path)
