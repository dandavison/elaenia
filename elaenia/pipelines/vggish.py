from sylph.learners.svm import SVMLearner
from sylph.pipeline import Compose
from sylph.pipeline import TrainingPipeline
from sylph.transforms.vggish import Audio2Spectrogram
from sylph.transforms.vggish import Spectrogram2VGGishEmbeddings


vggish_svm_learner_pipeline = TrainingPipeline(
    transform=Compose([Audio2Spectrogram(), Spectrogram2VGGishEmbeddings()]),
    learn=SVMLearner(),
)
