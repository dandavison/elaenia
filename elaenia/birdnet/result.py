from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Set

import pandas as pd

from sylph.audio import Audio


class BirdnetResult:
    true_species_ids: Dict[str, Set[int]] = {}
    NAMES = {
        "Eurasian Blackcap": "Blackcap",
        "Sylvia_atricapilla": "Blackcap",
        "Sylvia_borin": "Garden Warbler",
        "Greater Whitethroat": "Whitethroat",
        "Eurasian Blackbird": "Blackbird",
        "Common Chaffinch": "Chaffinch",
        "Common Nightingale": "Nightingale",
    }

    def __init__(self, path, recording: Optional[Audio] = None):
        self.path = Path(path)
        self.recording = recording
        self.df = pd.read_csv(path, sep="\t")
        for sp in ["Sylvia_atricapilla", "Sylvia_borin"]:
            self.true_species_ids[sp] = get_true_species_ids(sp)

    @property
    def species(self):
        return [self.NAMES.get(sp, sp) for sp in self.df["Common Name"]]

    @property
    def prob(self):
        return self.df["Confidence"]

    @property
    def species_to_probability(self):
        s2p = defaultdict(float)
        p_tot = 0.0
        for s, p in zip(self.species, self.prob):
            s2p[s] += p
            p_tot += p
        for s in s2p:
            s2p[s] = s2p[s] / p_tot
        return dict(s2p)

    @property
    def species_to_percent(self):
        return {k: 100 * v for k, v in self.species_to_probability.items()}

    @property
    def id(self):
        return int(self.path.name.split(".")[0])

    @property
    def url(self):
        return f"https://www.xeno-canto.org/{self.id}"

    @property
    def true_species(self):
        id = self.id
        [sp] = [sp for sp, ids in self.true_species_ids.items() if id in ids]
        return self.NAMES.get(sp, sp)

    def df_to_html(self):
        return self.df.to_html()


def get_true_species_ids(species) -> Set[int]:
    return {int(p.name.split(".")[0]) for p in Path(species).glob("*.wav")}
