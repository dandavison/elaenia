from elaenia.pipelines.vggish import vggish_svc_learner_pipeline
from elaenia.tests.datasets import TwoPureTonesDataset


def test_two_pure_tones():
    pipeline = vggish_svc_learner_pipeline
    dataset = TwoPureTonesDataset(n=20, duration=10, training_proportion=0.8)
    output = pipeline.run(dataset)
    metrics = pipeline.get_metrics(dataset, output)
    assert metrics["transformed_testing_accuracy"] == 1.0
    assert metrics["testing_accuracy"] == 1.0
