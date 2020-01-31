import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import elaenia.plot
from elaenia import librosa_utils


class Recording:
    def __init__(self):
        self.time_series = None
        self.sampling_rate = None

        # TODO: this need not be mp3
        self._mp3_file = None

    @property
    def mp3_file(self):
        return Path(self._mp3_file)

    @classmethod
    def from_file(cls, path):
        self = cls()
        self._mp3_file = Path(path)
        return self

    def load(self):
        if not self.mp3_file.exists():
            self.fetch_mp3()
        self.time_series, self.sampling_rate = librosa_utils.load(self.mp3_file, sr=None)

    def fetch_mp3(self):
        raise NotImplementedError

    def plot_spectrogram(self, ax=None):
        ax = ax or plt.gca()
        ss = elaenia.plot.stft(self.time_series)
        elaenia.plot.plot_spectrogram(ss, ax=ax, sr=self.sampling_rate)
        ax.set_title(self.mp3_file.name)


class NIPS4BPlusRecording(Recording):
    pass


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
    def mp3_file(self):
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
    def mp3_file(self):
        return Path(MP3_DIR, f"{self.id}.mp3")

    def fetch_mp3(self):
        print(f"{self.mp3_url} => {self.mp3_file}")
        with self.mp3_file.open("wb") as fp:
            resp = requests.get(self.mp3_url)
            resp.raise_for_status()
            fp.write(resp.content)
