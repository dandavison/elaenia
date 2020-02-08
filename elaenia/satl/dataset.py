import re
from collections import Counter
from functools import cached_property

import numpy as np

from elaenia.satl.experiment_recording import ExperimentRecording
from elaenia.satl.results import Results
from elaenia.utils import print_counter


class Dataset:
    def __init__(self, paths_file, experiment):
        self.paths_file = paths_file
        (self.name,) = re.match(r"(train|test)\.txt", paths_file.name).groups()
        self.experiment = experiment
        self.results = self.get_results()

    def get_results(self):
        results = Results(self._results_path, self)
        self._sanity_check(results)
        self._set_results_on_recordings(results)
        return results

    def _sanity_check(self, results):
        n_frames_in_dataset = len(self.frame_recording_ids)
        n_recordings_in_dataset = len(set(self.frame_recording_ids))
        assert len(self.recordings) == n_recordings_in_dataset
        assert len(results.recordings_predicted_integer_labels) == n_recordings_in_dataset
        assert len(results.frames_predicted_integer_labels) == n_frames_in_dataset

    def _set_results_on_recordings(self, results):
        for recording in self.recordings:
            recording.frames_predicted_integer_labels = results.frames_predicted_integer_labels[
                self.frame_recording_ids == recording.id
            ]
            recording.predicted_integer_label = results.recordings_predicted_integer_labels[
                recording.index
            ]

    @property
    def _results_path(self):
        paths = list(
            self.experiment.experiments_dir.glob(
                f"{self.name}_predictions_{self.experiment.name}*"
            )
        )
        if len(paths) > 1:
            raise AssertionError(f"Multiple results files: {sorted(paths)}")
        [path] = paths
        return path

    @cached_property
    def recordings(self):
        recordings = [
            ExperimentRecording.from_file(path, i, self)
            for i, path in enumerate(self.recording_paths)
        ]
        assert (s1 := set(r.id for r in recordings)) == (  # noqa:E203,E231
            s2 := set(self.frame_recording_ids)  # noqa:E203,E231
        ), (s1, s2)
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
    def frame_recording_ids(self):
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
