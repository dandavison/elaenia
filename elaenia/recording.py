import csv
import json
import re
import subprocess
import sys
import warnings
from functools import cached_property
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import requests

import elaenia.plot
import elaenia.stft
from elaenia import librosa_utils


class Recording:
    def __init__(self):
        self._time_series = None
        self._sampling_rate = None
        self._audio_file = None

    @property
    def audio_file(self):
        assert self._audio_file
        return Path(self._audio_file)

    @property
    def time_series(self):
        "1D numpy array of amplitudes"
        if self._time_series is None:
            self._load()
        return self._time_series

    @property
    def sampling_rate(self):
        "Hz"
        if self._sampling_rate is None:
            self._load()
        return self._sampling_rate

    @property
    def duration(self):
        "Seconds"
        return len(self.time_series) / self.sampling_rate

    @classmethod
    def from_file(cls, path):
        self = cls()
        self._audio_file = Path(path)
        return self

    def _load(self):
        self._time_series, self._sampling_rate = librosa_utils.load(self.audio_file, sr=None)

    def plot_spectrogram(self, n_fft, ax=None):
        ax = ax or plt.gca()
        ss = elaenia.stft.stft(self.time_series, n_fft=n_fft)
        elaenia.plot.plot_spectrogram(ss, ax=ax, sr=self.sampling_rate)
        ax.set_title(self.audio_file.name)
        return ss

    def play(self):
        subprocess.check_call(["open", "-a", "/Applications/Cog.app", self.audio_file])


class NIPS4BPlusRecording(Recording):
    ROOT_DIR = Path("/tmp/NIPS4Bplus")

    def __init__(self, id, dataset="train"):
        super().__init__()
        self.id = str(id)
        self.dataset = dataset
        self._temporal_annotations = None

    @classmethod
    def from_file(cls, file_name):
        file_name = Path(file_name).name
        id, = re.match(r"^nips4b_birds_trainfile([0-9]+)\.wav$", file_name).groups()
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
    def temporal_annotations(self):
        if not self._temporal_annotations:
            path = (
                self.ROOT_DIR
                / "temporal_annotations_nips4b"
                / f"annotation_{self.dataset}{self.id}.csv"
            )
            annotations = []
            with open(path) as fp:
                for row in csv.DictReader(fp, fieldnames=["start", "duration", "label"]):
                    for field in ["start", "duration"]:
                        row[field] = float(row[field])
                    annotations.append(row)
            self._temporal_annotations = annotations
        return self._temporal_annotations

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
                r"^([0-9]+) ([0-9]+) (.+) ([0-9]+) .*\.mp3$", file_name
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


class XenoCantoRecording0(Recording):
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
        return self.MP3_DIR / f"{self.id}.mp3"

    def _load(self):
        if not self.audio_file.exists():
            self._fetch_mp3()
        super()._load()

    def _fetch_mp3(self):
        print(f"{self.mp3_url} => {self.audio_file}")
        with self.audio_file.open("wb") as fp:
            resp = requests.get(self.mp3_url)
            resp.raise_for_status()
            fp.write(resp.content)


class XenoCantoRecording(Recording):
    """
    A recording downloaded using https://github.com/ntivirikin/xeno-canto-py.
    """

    ROOT_DIR = Path("/tmp/xeno-canto-data")

    # The name given to the directory under metadata/, e.g. "gen_Xiphorhynchus"
    # This class is abstract; concrete subclasses must set this.
    QUERY = None

    def __init__(self, id: int):
        """
        id    -- the integer ID (i.e. without the XC prefix)
        """
        self.id = int(id)

    @classmethod
    def for_species(cls, species: Tuple[str, str]):
        assert len(species) == 2
        return [
            cls(rec["id"])
            for page in cls._api_response_pages()
            for rec in page["recordings"]
            if (rec["gen"], rec["sp"]) == species
        ]

    @property
    def audio_file(self):
        species = self.metadata["en"].replace(" ", "")
        return self.ROOT_DIR / "audio" / species / f"{self.id}.mp3"

    def is_song(self):
        type_field = self.metadata["type"]
        type_field = type_field.lower()
        return "song" in type_field and "call" not in type_field

    @cached_property
    def metadata(self):
        return next(
            rec
            for page in self._api_response_pages()
            for rec in page["recordings"]
            if rec["id"] == str(self.id)
        )

    @classmethod
    def _api_response_pages(cls):
        paths = (cls.ROOT_DIR / "metadata" / cls.QUERY).glob("*.json")
        for path in paths:
            with open(path) as fp:
                yield json.load(fp)
