import re
from collections import Counter
from functools import cached_property

import numpy as np

from elaenia.recording import Recording
from elaenia.utils import print_counter


class DatasetRecording(Recording):
    @classmethod
    def from_file(cls, path, dataset):
        self = super().from_file(path)
        self.dataset = dataset
        return self

    @property
    def label(self):
        return str(self.audio_file.relative_to(self.audio_file.parents[1]))

    @property
    def frame_vggish_embeddings(self):
        return self.dataset.frame_vggish_embeddings[
            self.dataset.frame_recording_labels == self.label
        ]


class Dataset:
    def __init__(self, paths_file, experiment):
        self.paths_file = paths_file
        self.name, = re.match(r"(train|test)\.txt", paths_file.name).groups()
        self.experiment = experiment

    @cached_property
    def recordings(self):
        recordings = [DatasetRecording.from_file(path,  self) for path in self.recording_paths]
        assert (s1 := set(r.label for r in recordings)) == (s2 := set(self.frame_recording_labels)), (s1, s2)
        return recordings

    @property
    def recording_paths(self):
        paths = []
        with open(self.paths_file) as fp:
            for line in fp:
                path = self.experiment.audio_dir / line.rstrip()
                assert path.exists(), path
                paths.append(path)
        return paths

    @property
    def frame_vggish_embeddings(self):
        """
        (n_frames, 128) array: the pre-trained VGGish network computes a 128-dimensional embedding
        vector for each 0.96 s frame.

        Note that the following should be approximately equal:

        >>> experiment.train_set.frame_vggish_embeddings.shape[0]
        >>> sum(r.duration for r in experiment.train_set.recordings) / 0.96
        """
        return self._frame_data["X"]

    @property
    def frame_class_labels(self):
        """
        (nframes,) array of integer class labels.
        """
        return self._frame_data["Y"]

    @property
    def frame_recording_labels(self):
        """
        (nframes,) array of elements like "Xiphorhynchus_elegans/249864.mp3"
        """
        return self._frame_data["IDS"]

    @cached_property
    def _frame_data(self):
        prefix = {"train": "training_data", "test": "evaluation_data"}[self.name]
        path = (
            self.experiment.audio_representations_dir
            / f"{prefix}_{self.experiment.name}_vggish.npz"
        )
        return np.load(path)

    def count_classes(self):
        with open(self.paths_file) as fp:
            return Counter(_class for line in fp for _class, file_name in [line.split("/")])

    def describe(self):
        print(self.name)
        print_counter(self.count_classes())
