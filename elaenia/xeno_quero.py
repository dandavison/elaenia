import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from typing import Tuple

from cached_property import cached_property

from sylph.audio import Audio


class XenoQueroRecording:
    class PathDoesNotExist(AssertionError):
        pass

    def __init__(self, id: int):
        self.id = int(id)

    @cached_property
    def metadata(self):
        with Path(self.get_metadata_path()).open() as fp:
            return json.load(fp)

    @property
    def has_audio(self) -> bool:
        path = self.get_recording_path()
        return path and path.exists()

    @property
    def audio(self) -> Audio:
        return Audio.from_file(self.get_recording_path())

    @property
    def country(self):
        return self.metadata.get("cnt", "(No country)")

    @property
    def location(self):
        return self.metadata.get("loc", "(No location)")

    @property
    def coordinates(self) -> Optional[Tuple[float, float]]:
        try:
            return float(self.metadata["lat"]), float(self.metadata["lng"])
        except TypeError:
            return None

    @property
    def url(self):
        return f"https://www.xeno-canto.org/{self.id}"

    @property
    def genus(self):
        return self.metadata.get("gen")

    @property
    def species(self):
        return self.metadata.get("sp")

    @property
    def english_name(self):
        return self.metadata.get("en", "(No English species name)")

    @property
    def scientific_name(self):
        genus = self.genus or "(No genus)"
        species = self.species or "(No species)"
        return f"{genus} {species}"

    @property
    def is_song(self):
        type_words = self.metadata.get("type", "").lower().split()
        return "song" in type_words and "call" not in type_words

    def get_metadata_path(self):
        try:
            return self._get_path_from_id("json")
        except self.PathDoesNotExist:
            raise self.PathDoesNotExist("No recording data found for xeno-canto id {self.id}")

    def get_recording_path(self):
        try:
            return self._get_path_from_id("mp3")
        except self.PathDoesNotExist:
            return None

    def _get_path_from_id(self, suffix: str):
        assert not suffix.startswith(".")
        dir = get_download_directory()
        paths = list(dir.glob(f"*/{self.id}.{suffix}"))
        if not paths:
            raise self.PathDoesNotExist
        [path] = paths
        return Path(path)


def get_download_directory():
    dir = os.getenv("XENO_QUERO_DONWLOAD_DIRECTORY", "").strip()
    if not dir:
        raise AssertionError("XENO_QUERO_DONWLOAD_DIRECTORY env var is not set.")
    dir = Path(dir)
    if not dir.is_dir():
        raise AssertionError(f"{dir} is not a directory.")
    return dir
