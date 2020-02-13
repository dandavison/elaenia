from elaenia.pipeline import Compose
from elaenia.pipeline import Pipeline
from elaenia.transforms.vggish import Audio2Spectrogram
from elaenia.transforms.vggish import Spectrogram2VGGishEmbeddings
from elaenia.transforms.sklearn import SVCLearner

vggish_svc_learner_pipeline = Pipeline(
    transform=Compose([Audio2Spectrogram(), Spectrogram2VGGishEmbeddings()]), learn=SVCLearner()
)
