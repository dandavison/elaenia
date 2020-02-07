import importlib
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")

import ipdb
import matplotlib.pyplot as plt
import numpy as np


import elaenia.satl.plot
from elaenia import plot
from elaenia import recording
from elaenia import satl
from elaenia import stft


def reload():
    importlib.reload(plot)
    importlib.reload(recording)
    importlib.reload(stft)
    importlib.reload(satl.dataset)
    importlib.reload(satl.experiment)
    importlib.reload(satl.plot)

    recording.elaenia.plot = plot
    satl.dataset.Recording = recording.Recording


plt.ion()
experiment_name = "Xiphorhynchus_guttatus_vs_elegans"
