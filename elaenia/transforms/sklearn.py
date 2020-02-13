from dataclasses import dataclass

import numpy as np

import sklearn.svm
import sklearn.base

from elaenia.classifier import Classifier
from elaenia.dataset import Dataset
from elaenia.pipeline import Learner


@dataclass
class SKClassifier(Classifier):
    """
    A wrapper around a trained sklearn classifier, used for prediction.
    """

    model: sklearn.base.ClassifierMixin

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(observations)


class SKClassifierLearner(Learner):
    """
    Takes a Dataset and outputs a SKClassifier.
    """

    model: sklearn.base.ClassifierMixin

    def __call__(self, dataset: Dataset) -> SKClassifier:
        X, y = dataset.observations, dataset.labels
        X, y = map(dataset.unpack_objects, (X, y))
        model = self.model.fit(X, y)
        return SKClassifier(model=model)


class SVCLearner(SKClassifierLearner):
    model = sklearn.svm.SVC()
