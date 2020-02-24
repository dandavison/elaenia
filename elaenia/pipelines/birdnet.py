from pathlib import Path

from sylph.learners.svm import SVMLearner
from sylph.pipeline import Compose
from sylph.pipeline import TrainingPipeline
from sylph.transforms.birdnet import Audio2Embeddings


def make_birdnet_embeddings_training_pipeline(birdnet_output_dir=None):
    return TrainingPipeline(
        transform=Compose([Audio2Embeddings(birdnet_output_dir=birdnet_output_dir)]),
        learn=SVMLearner(),
    )
