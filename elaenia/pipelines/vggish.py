from sylph.learners.svm import SVMLearner
from sylph.pipeline import Compose
from sylph.pipeline import TrainingPipeline
from sylph.transforms.audio import Audio2Audio16Bit
from sylph.transforms.pca import PCA
from sylph.transforms.vggish import Audio2Spectrogram
from sylph.transforms.vggish import Spectrogram2VGGishEmbeddings


vggish_svm_learner_pipeline = TrainingPipeline(
    transform=Compose(
        [Audio2Audio16Bit(normalize_amplitude=True), Audio2Spectrogram(), Spectrogram2VGGishEmbeddings()]
    ),
    learn=SVMLearner(),
)

vggish_pca_svm_learner_pipeline = TrainingPipeline(
    transform=Compose(
        [
            Audio2Audio16Bit(normalize_amplitude=True),
            Audio2Spectrogram(),
            Spectrogram2VGGishEmbeddings(),
            PCA(whiten=True),
        ]
    ),
    learn=SVMLearner(),
)
