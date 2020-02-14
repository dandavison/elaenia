import numpy as np
from sklearn.metrics import accuracy_score

from elaenia.pipeline import Compose
from elaenia.pipeline import TrainingPipeline
from elaenia.transforms.vggish import Audio2Spectrogram
from elaenia.transforms.vggish import Spectrogram2VGGishEmbeddings
from elaenia.transforms.sklearn import SVCLearner
from elaenia.utils import get_group_modes


class VGGishTrainingPipeline(TrainingPipeline):
    def get_metrics(self, dataset, output):
        testing_dataset_predictions = np.array(
            get_group_modes(
                output["transformed_testing_dataset_predictions"],
                output["transformed_testing_dataset"].n_examples_per_audio,
            )
        )
        return {
            "testing_dataset_predictions": testing_dataset_predictions,
            "transformed_testing_accuracy": accuracy_score(
                output["transformed_testing_dataset"].labels,
                output["transformed_testing_dataset_predictions"],
            ),
            "testing_accuracy": accuracy_score(
                dataset.testing_dataset.labels, testing_dataset_predictions
            ),
        }


vggish_svc_learner_pipeline = VGGishTrainingPipeline(
    transform=Compose([Audio2Spectrogram(), Spectrogram2VGGishEmbeddings()]), learn=SVCLearner()
)
