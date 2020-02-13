from elaenia.pipeline import LearnerPipeline
from elaenia.transforms.vggish import Audio2VGGishEmbeddings
from elaenia.transforms.sklearn import SVCLearner

vggish_svc_learner_pipeline = LearnerPipeline(Audio2VGGishEmbeddings(), learner=SVCLearner())
