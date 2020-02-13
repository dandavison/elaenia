"""
The interface of this module is intended to be compatible with torchvision.Transform and
torchvision.Compose.
"""
from typing import List
from typing import Union
from typing_extensions import Protocol

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
    def __init__(self, operations: List[Union[Transform, Learner]]):
        self.operations = operations

    def __call__(self, x: Dataset) -> Union[Dataset, Classifier]:
        for operation in self.operations:
            x = operation(x)  # type: ignore
        return x
