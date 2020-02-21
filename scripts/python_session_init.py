import importlib
from collections import Counter
from pathlib import Path

import matplotlib

if False:
    matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np


from elaenia import plot
from elaenia import recording
from elaenia import stft
from elaenia import vggish


def reload():
    importlib.reload(plot)
    importlib.reload(recording)
    importlib.reload(stft)
    importlib.reload(vggish)

    recording.elaenia.plot = plot


plt.ion()
experiment_name = "Xiphorhynchus_guttatus_vs_elegans"
