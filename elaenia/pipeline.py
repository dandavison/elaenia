"""
The interface of this module is intended to be compatible with torchvision.Transform and
torchvision.Compose.
"""
from typing import List
from typing import Protocol
from typing import Union

from elaenia.classifier import Classifier
from elaenia.dataset import Dataset


class Transform(Protocol):
    """
    A Transform takes in one Dataset and outputs another.
    """

    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError


class Learner(Protocol):
    """
    A Learner takes in a Dataset and outputs a Classifier.
    """

    def __call__(self, dataset: Dataset) -> Classifier:
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: List[Transform], learner: Learner = None):
        self.transforms = transforms
        self.learner = learner

    def __call__(self, dataset: Dataset) -> Union[Dataset, Classifier]:
        for transform in self.transforms:
            dataset = transform(dataset)
        if not self.learner:
            return dataset
        else:
            return self.learner(dataset)
