import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import elaenia.plot
import elaenia.stft
from elaenia import librosa_utils


class Recording:
    def __init__(self):
        self.time_series = None
        self.sampling_rate = None

        self._audio_file = None

    @property
    def audio_file(self):
        return Path(self._audio_file)

    @classmethod
    def from_file(cls, path):
        self = cls()
        self._audio_file = Path(path)
        return self

    def load(self):
        if not self.audio_file.exists():
            self.fetch_mp3()
        self.time_series, self.sampling_rate = librosa_utils.load(self.audio_file, sr=None)

    def fetch_mp3(self):
        raise NotImplementedError

    def plot_spectrogram(self, n_fft, ax=None):
        ax = ax or plt.gca()
        ss = elaenia.stft.stft(self.time_series, n_fft=n_fft)
        elaenia.plot.plot_spectrogram(ss, ax=ax, sr=self.sampling_rate)
        ax.set_title(self.audio_file.name)


class NIPS4BPlusRecording(Recording):
    ROOT_DIR = Path("/tmp/NIPS4Bplus")

    def __init__(self, id, dataset="train"):
        super().__init__()
        self.id = str(id)
        self.dataset = dataset
        self.temporal_annotations = None

    @classmethod
    def from_file(cls, file_name):
        file_name = Path(file_name).name
        id, = re.match("^nips4b_birds_trainfile([0-9]+)\.wav$", file_name).groups()
        return cls(id, dataset="train")

    @property
    def audio_file(self):
        return (
            self.ROOT_DIR
            / "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"
            / self.dataset
            / f"nips4b_birds_trainfile{self.id}.wav"
        )

    @property
    def temporal_annotations_file(self):
        return (
            self.ROOT_DIR
            / "temporal_annotations_nips4b"
            / f"annotation_{self.dataset}{self.id}.csv"
        )

    def load(self):
        super().load()
        self._load_temporal_annotations()

    def _load_temporal_annotations(self):
        self.temporal_annotations = []
        with open(self.temporal_annotations_file) as fp:
            for row in csv.DictReader(fp, fieldnames=["start", "duration", "label"]):
                for field in ["start", "duration"]:
                    row[field] = float(row[field])
                self.temporal_annotations.append(row)

    def plot_spectrogram(self, **kwargs):
        super().plot_spectrogram(**kwargs)
        cols = ["g", "r"]
        for i, row in enumerate(self.temporal_annotations):
            xx = [row["start"], row["start"] + row["duration"]]
            plt.vlines(xx, colors=cols, linestyles=":", ymin=0, ymax=3e4)
            for x, col in zip(xx, cols):
                plt.text(
                    x,
                    -2500,
                    f"{row['label']} {i}",
                    color=col,
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                )


class BoesmanRecording(Recording):
    MP3_DIR = Path("/tmp/boesman-mp3s")

    def __init__(self, species_id, recording_id, english_name=None):
        super().__init__()
        self.species_id = species_id
        self.recording_id = recording_id

    @classmethod
    def from_file(cls, file_name):
        return cls(**cls.parse_file_name(file_name))

    @classmethod
    def from_english_name(cls, english_name):
        return [
            cls.from_file(file_name)
            for file_name in cls.MP3_DIR.glob("*.mp3")
            if cls.file_name_match(file_name, english_name)
        ]

    @property
    def same_species_recordings(self):
        return [
            type(self).from_file(file_name)
            for file_name in self.MP3_DIR.glob(f"{self.species_id} *")
        ]

    @staticmethod
    def file_name_match(file_name, english_name):
        file_name = str(file_name)

        def transform(s):
            return s.replace("-", " ").lower()

        return transform(english_name) in transform(file_name)

    @staticmethod
    def parse_file_name(file_name):
        file_name = Path(file_name).name
        try:
            species_id, recording_id, english_name, recording_id_2 = re.match(
                "^([0-9]+) ([0-9]+) (.+) ([0-9]+) .*\.mp3$", file_name
            ).groups()
        except Exception:
            sys.stderr.write("parse_file_name error\n")
            sys.stderr.write(file_name + "\n")
            raise

        if recording_id != recording_id_2:
            warnings.warn(f"Recording IDs differ in file name: {file_name}")
        return {
            "species_id": species_id,
            "recording_id": recording_id,
            "english_name": english_name,
        }

    @property
    def audio_file(self):
        file, = self.MP3_DIR.glob(f"{self.species_id} {self.recording_id} *")
        return file


class XCRecording(Recording):
    MP3_DIR = Path("/tmp/xenocanto-mp3s")

    def __init__(self, id):
        super().__init__()
        assert re.match("XC[0-9]+", id)
        self.id = id
        self.MP3_DIR.mkdir(exist_ok=True, parents=True)

    @property
    def mp3_url(self):
        return f"https://www.xeno-canto.org/{id[2:]}/download"

    @property
    def audio_file(self):
        return MP3_DIR / f"{self.id}.mp3"

    def fetch_mp3(self):
        print(f"{self.mp3_url} => {self.audio_file}")
        with self.audio_file.open("wb") as fp:
            resp = requests.get(self.mp3_url)
            resp.raise_for_status()
            fp.write(resp.content)
