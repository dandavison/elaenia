import importlib
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")

import ipdb
import matplotlib.pyplot as plt
import numpy as np


import elaenia.satl.dataset
import elaenia.satl.experiment
import elaenia.satl.plot
import elaenia.satl.results
from elaenia import plot
from elaenia import recording
from elaenia import satl
from elaenia import stft
from elaenia import vggish


def reload():
    importlib.reload(plot)
    importlib.reload(recording)
    importlib.reload(stft)
    importlib.reload(satl.dataset)
    importlib.reload(satl.experiment)
    importlib.reload(satl.experiment_recording)
    importlib.reload(satl.plot)
    importlib.reload(satl.results)
    importlib.reload(vggish)

    recording.elaenia.plot = plot
    satl.dataset.ExperimentRecording = satl.experiment_recording.ExperimentRecording
    satl.dataset.Recording = recording.Recording
    satl.dataset.Results = satl.results.Results
    satl.experiment.Results = satl.results.Results
    satl.plot.ExperimentRecording = satl.experiment_recording.ExperimentRecording
    satl.plot.VGGishFrames = vggish.VGGishFrames


plt.ion()
experiment_name = "Xiphorhynchus_guttatus_vs_elegans"
