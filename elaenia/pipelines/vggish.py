from elaenia.pipeline import Compose
from elaenia.transforms.vggish import Audio2VGGishEmbeddings
from elaenia.transforms.sklearn import SVCLearner

vggish_svc_learner_pipeline = Compose([Audio2VGGishEmbeddings(), SVCLearner()])
