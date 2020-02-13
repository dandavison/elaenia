from elaenia.pipelines.vggish import vggish_svc_learner_pipeline
from elaenia.tests.datasets import TwoPureTonesDataset


def test_two_pure_tones():
    pipeline = vggish_svc_learner_pipeline
    dataset = TwoPureTonesDataset(n=20, duration=10)
    classifier = pipeline.learn(dataset.training_dataset)
    test_predictions = classifier.predict(dataset.testing_dataset.observations)
    assert dataset.testing_dataset.compute_accuracy(test_predictions) == 1.0
