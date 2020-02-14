import numpy as np
from sklearn.metrics import accuracy_score

from sylph.learners.svm import SVMLearner
from sylph.pipeline import Compose
from sylph.pipeline import TrainingPipeline
from sylph.transforms.pca import PCA
from sylph.transforms.vggish import Audio2Spectrogram
from sylph.transforms.vggish import Spectrogram2VGGishEmbeddings
from sylph.utils import get_group_modes


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


vggish_svm_learner_pipeline = VGGishTrainingPipeline(
    transform=Compose([Audio2Spectrogram(), Spectrogram2VGGishEmbeddings(), PCA(whiten=True)]),
    learn=SVMLearner(),
)
