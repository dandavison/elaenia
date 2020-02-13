"""
The interface of this module is intended to be compatible with torchvision.Transform and
torchvision.Compose.
"""
from dataclasses import dataclass
from typing import List
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
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, dataset: Dataset) -> Dataset:
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset


@dataclass
class Pipeline:
    transform: Transform
    learn: Learner
