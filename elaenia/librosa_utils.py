import warnings

import librosa


def load(file, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.load(file, **kwargs)
