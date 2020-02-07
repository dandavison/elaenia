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
