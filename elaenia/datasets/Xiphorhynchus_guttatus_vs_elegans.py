from pathlib import Path

import numpy as np

from sylph.audio_dataset import AudioDataSet


def get_paths_and_labels(root_dir, dataset_name, train_or_test):
    """
    Read a file that contains lines looking like
    Xiphorhynchus_elegans/264573.mp3
    """
    paths_file = root_dir / "index" / dataset_name / f"{train_or_test}.txt"
    audio_dir = root_dir / "audio" / dataset_name
    species_and_file_names = [line.rstrip().split("/") for line in paths_file.open()]
    paths = [
        audio_dir / species_name / f"{file_name}"
        for (species_name, file_name) in species_and_file_names
    ]
    labels = [species_name for (species_name, file_name) in species_and_file_names]
    return paths, labels


class Xiphorhynchus_guttatus_vs_elegans_DataSet(AudioDataSet):
    @classmethod
    def create(cls):
        root_dir = Path("~/src/3p/sklearn-audio-transfer-learning/data").expanduser()
        dataset_name = "Xiphorhynchus_guttatus_vs_elegans"
        train_paths, train_labels = get_paths_and_labels(root_dir, dataset_name, "train")
        test_paths, test_labels = get_paths_and_labels(root_dir, dataset_name, "test")
        paths = np.array(train_paths + test_paths)
        labels = np.array(train_labels + test_labels)
        training_rows = np.array([i < len(train_paths) for i in range(len(paths))])
        return cls.from_files(paths, labels, training_rows=training_rows)
