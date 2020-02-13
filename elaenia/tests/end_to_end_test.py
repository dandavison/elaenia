import numpy as np
from sklearn.metrics import accuracy_score

from elaenia.pipelines.vggish import vggish_svc_learner_pipeline
from elaenia.tests.datasets import TwoPureTonesDataset
from elaenia.utils import get_group_modes


def test_two_pure_tones():
    pipeline = vggish_svc_learner_pipeline
    dataset = TwoPureTonesDataset(n=100, duration=10)
    _assert_perfect_predictions(pipeline, dataset)


def _assert_perfect_predictions(pipeline, dataset):
    transformed_training_dataset = pipeline.transform(dataset.training_dataset)
    classifier = pipeline.learn(transformed_training_dataset)
    transformed_testing_dataset = pipeline.transform(dataset.testing_dataset)
    transformed_testing_dataset_predictions = classifier.predict(
        transformed_testing_dataset.observations
    )
    assert (
        accuracy_score(transformed_testing_dataset.labels, transformed_testing_dataset_predictions)
        == 1.0
    )
    testing_dataset_predictions = np.array(
        get_group_modes(
            transformed_testing_dataset_predictions,
            transformed_testing_dataset.n_examples_per_audio,
        )
    )
    assert accuracy_score(dataset.testing_dataset.labels, testing_dataset_predictions) == 1.0
