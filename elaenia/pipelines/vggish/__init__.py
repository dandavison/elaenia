from elaenia.pipeline import Compose
from elaenia.transforms.vggish import Audio2Spectrogram
from elaenia.transforms.vggish import Spectrogram2VGGishEmbeddings
from elaenia.transforms.sklearn import SVCLearner

vggish_svc_learner_pipeline = Compose([Audio2Spectrogram(), Spectrogram2VGGishEmbeddings(), SVCLearner()])
