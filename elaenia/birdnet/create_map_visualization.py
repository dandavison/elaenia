import sys
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import List

import folium
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins.beautify_icon import BeautifyIcon

from elaenia.birdnet import BirdnetResult
from elaenia.xeno_quero import XenoQueroRecording


@dataclass
class Results:
    results: List[BirdnetResult]

    def __iter__(self):
        return iter(self.results)

    def coordinates_centroid(self):
        assert len(self.results)
        coordinates = np.array([r.recording.coordinates for r in self])
        return tuple(coordinates.mean(axis=0))


def get_results(birdnet_output_paths):
    xc_ids = [int(p.name.split(".")[0]) for p in birdnet_output_paths]
    recordings = list(map(XenoQueroRecording, xc_ids))
    results = list(starmap(BirdnetResult, zip(birdnet_output_paths, recordings)))
    results_with_coordinates = [r for r in results if r.recording.coordinates]
    n_without_coordinates = len(results_with_coordinates) - len(results)
    if n_without_coordinates:
        print(f"Excluding {n_without_coordinates} recordings without coordinates")
        results = results_with_coordinates
    return Results([r for r in results if r.recording.has_audio])


def create_map(results):
    m = folium.Map(results.coordinates_centroid(), zoom_start=4)
    nn_coords = [NeuralNetCoordinate(r) for r in results]
    nn_coord_coordinates = sorted(x.coordinate for x in nn_coords if x.coordinate is not None)
    colormap = plt.get_cmap("coolwarm")
    colors = colormap(nn_coord_coordinates)
    species_to_icon = {"polyglotta": "circle", "icterina": "circle"}
    species_to_border_color = {"polyglotta": rgb2hex(colors[-1]), "icterina": rgb2hex(colors[0])}
    for r, nn_coord in zip(results, nn_coords):

        if nn_coord.coordinate is None:
            continue

        if nn_coord.melwar_prob < 0.5 and nn_coord.ictwar_prob < 0.5:
            continue

        if not r.recording.is_song:
            continue

        icon_shape = species_to_icon[r.recording.species]
        border_color = species_to_border_color[r.recording.species]
        color = rgb2hex(colors[nn_coord_coordinates.index(nn_coord.coordinate)])
        if False:
            icon = folium.Icon(color="white", icon_color=color)
        else:
            icon = BeautifyIcon(
                icon="", icon_shape=icon_shape, background_color=color, border_color=border_color
            )

        tooltip = (
            f"{nn_coord.coordinate:.2f} "
            f"({nn_coord.melwar_prob:.2f}, {nn_coord.ictwar_prob:.2f})"
            f": {r.recording.english_name}"
        )
        popup = make_popup(r)

        folium.Marker(r.recording.coordinates, icon=icon, tooltip=tooltip, popup=popup).add_to(m)
    return m


@dataclass
class NeuralNetCoordinate:
    result: BirdnetResult

    @property
    def coordinate(self):
        denom = self.melwar_prob + self.ictwar_prob
        return self.melwar_prob / denom if denom else None

    @property
    def melwar_prob(self):
        return self.result.species_to_probability.get("Melodious Warbler", 0.0)

    @property
    def ictwar_prob(self):
        return self.result.species_to_probability.get("Icterine Warbler", 0.0)


def make_popup(result):
    rec = result.recording
    return f"""
    <a href="{rec.url}">{rec.english_name} <i>{rec.scientific_name}</i></a>
    <p>{rec.country}: {rec.location} {rec.coordinates}<p>
    <p>{rec.metadata.get('rmk', '')}</p>
    {result.df_to_html()}
    """


if __name__ == "__main__":
    # paths look like 397043.mp3.BirdNET.selections.txt
    birdnet_output_paths = list(map(Path, sys.argv[1:]))
    results = get_results(birdnet_output_paths)
    m = create_map(results)
    m.save("map.html")
