import os
from pathlib import Path


def getenv(key, default=None):
    value = os.getenv(key, default=default)
    if not value:
        raise AssertionError(f"Environment variable not set: {key}")
    return value


def get_egg_path():
    dir = Path(__file__).resolve().parents[2] / "dist"
    try:
        [path] = dir.glob("elaenia-*.egg")
    except ValueError:
        raise AssertionError(f"Failed to find egg file in directory: {dir}")
    else:
        return path


def print_counter(cnts):
    width = max(len(k) for k in cnts)
    for k, n in sorted(cnts.items()):
        print(f"{k:{width}} {n}")


def delete_directory_tree(dir):
    if dir.exists():
        for f in dir.glob("*"):
            if f.is_dir():
                delete_directory_tree(f)
            else:
                f.unlink()
        dir.rmdir()


def split(it, proportion):
    it = list(it)
    n = int(len(it) * proportion)
    return it[:n], it[n:]
