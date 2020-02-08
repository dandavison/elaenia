import pprint
import random
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import List
from typing import Tuple

from elaenia.recording import XenoCantoRecording
from elaenia.satl.dataset import Dataset
from elaenia.satl.results import Results
from elaenia.utils import delete_directory_tree
from elaenia.utils import split

DATA_DIR = Path("/tmp/satl-data")
RANDOMIZE_DATA = False


class Experiment:
    def __init__(self, name):
        self.name = name

        self.audio_dir = DATA_DIR / "audio" / name
        self.audio_representations_dir = DATA_DIR / "audio_representations"
        self.experiments_dir = DATA_DIR / "experiments"

        self.train_set = Dataset(DATA_DIR / "index" / name / "train.txt", self)
        self.test_set = Dataset(DATA_DIR / "index" / name / "test.txt", self)
        self.test_set.results = self.test_set.get_results()

    def describe(self):
        self.train_set.describe()
        print()
        self.test_set.describe()


def create_training_experiment(
    dataset,
    recording_cls: XenoCantoRecording,
    species: List[Tuple[str, str]],
    data_dir,
    song=True,
    train_proportion=0.8,
):
    """
    https://github.com/jordipons/sklearn-audio-transfer-learning
    """
    data_dir = Path(data_dir).expanduser()
    assert data_dir.exists(), data_dir
    audio_symlink_dir = data_dir / "audio" / dataset
    paths_files_dir = data_dir / "index" / dataset
    for dir in [audio_symlink_dir, paths_files_dir]:
        delete_directory_tree(dir)
        dir.mkdir(parents=True)
    train_paths_file = paths_files_dir / "train.txt"
    test_paths_file = paths_files_dir / "test.txt"

    def species_label(genus, species):
        return f"{genus}_{species}"

    training_data = {sp: recording_cls.for_species(sp) for sp in species}
    # Discard data so that all labels have the same count
    min_label_count = min(len(recs) for recs in training_data.values())
    training_data = {sp: recs[:min_label_count] for sp, recs in training_data.items()}

    if RANDOMIZE_DATA:
        all_data = list(chain.from_iterable(training_data.values()))
        random.shuffle(all_data)
        for n, sp in enumerate(training_data):
            offset = n * min_label_count
            training_data[sp] = all_data[offset : (offset + min_label_count)]  # noqa:E203

    for sp in species:
        train, test = split(training_data[sp], train_proportion)
        species_audio_dir = audio_symlink_dir / species_label(*sp)
        delete_directory_tree(species_audio_dir)
        species_audio_dir.mkdir(parents=True)
        for recordings, paths_file in [(train, train_paths_file), (test, test_paths_file)]:
            with open(paths_file, "a") as fp:
                for rec in recordings:
                    symlink = species_audio_dir / rec.audio_file.name
                    symlink.symlink_to(rec.audio_file)
                    fp.write(f"{symlink.relative_to(audio_symlink_dir)}\n")

    path2gt_datasets = StringIO()
    path2gt_datasets.write(
        f"""
    if dataset == "{dataset}":
"""
    )
    for label, sp in enumerate(species):
        path2gt_datasets.write(
            f"""
        if path.startswith("{species_label(*sp)}/"):
            return {label}
"""
        )
    path2gt_datasets = path2gt_datasets.getvalue()

    config = {
        "dataset": dataset,
        "num_classes_dataset": len(species),
        "audio_folder": f"{audio_symlink_dir}/",
        "audio_paths_train": f"{train_paths_file}",
        "audio_paths_test": f"{test_paths_file}",
        "batch_size": 8,
        "features_type": "vggish",
        "pca": 128,
        "model_type": "SVM",
        "load_training_data": False,
        "load_evaluation_data": False,
    }
    print(path2gt_datasets)
    print()
    print(f"config = {pprint.pformat(config)}")
