from collections import Counter
from typing import List
from typing import TypeVar

T = TypeVar("T")


def get_group_modes(values: List[T], group_sizes: List[int]) -> List[T]:
    assert sum(group_sizes) == len(values)
    offset = 0
    modes = []
    for n in group_sizes:
        group = values[offset: (offset + n)]
        mode, cnt = Counter(group).most_common()[0]
        modes.append(mode)
        offset += n
    return modes


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
