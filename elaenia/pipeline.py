from elaenia.classifier import Classifier
from elaenia.dataset import Dataset


class Transformer:
    """
    A Transformer takes in one Dataset and outputs another.
    """

    def transform(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError


class Learner:
    """
    A Learner takes in a Dataset and outputs a Classifier.
    """

    def learn(self, dataset: Dataset) -> Classifier:
        raise NotImplementedError


class TransformerPipeline:
    """
    A TransformerPipeline takes in a Dataset, passes it through a sequence of Transformers, and
    outputs a Dataset.
    """

    def __init__(self, *transformers: Transformer):
        self.transformers = transformers

    def transform(self, dataset: Dataset) -> Dataset:
        for transformer in self.transformers:
            dataset = transformer.transform(dataset)
        return dataset


class LearnerPipeline(TransformerPipeline):
    """
    A LearnerPipeline takes in a Dataset, passes it through a sequence of Transformers to a
    Learner, and outputs a Classifier.
    """

    def __init__(self, *transformers: Transformer, learner: Learner):
        super().__init__(*transformers)
        self.learner = learner
        self._validate()

    def _validate(self):
        assert all(isinstance(t, Transformer) for t in self.transformers)
        assert isinstance(self.learner, Learner)

    def learn(self, dataset: Dataset) -> Classifier:
        dataset = self.transform(dataset)
        return self.learner.learn(dataset)
