from pathlib import Path

from elaenia.pipelines.birdnet import make_birdnet_embeddings_training_pipeline

from sylph.xeno_quero import get_download_directory
from sylph.xeno_quero.dataset import XenoQueroDataset


def make_dataset():
    """
    A standard two-species XenoQuero dataset, but excluding recordings for which embeddings don't exist.
    """
    training_proportion = 0.8
    dataset = XenoQueroDataset.from_species(
        ["Hippolais icterina", "Hippolais polyglotta"], training_proportion=training_proportion
    )
    good_rows = [have_embeddings(recording.id) for recording in dataset.observations]
    dataset = dataset.get_rows(good_rows)
    dataset.training_proportion = training_proportion
    return dataset


def have_embeddings(id):
    return bool(list((get_download_directory() / "Hippolais").glob(f"{id}-*-embedding.npz")))


if __name__ == "__main__":
    from clint.textui import colored

    red = lambda s: colored.red(s, bold=True)
    print(red("make_dataset"))

    dataset = make_dataset()
    pipeline = make_birdnet_embeddings_training_pipeline(
        birdnet_output_dir=Path("~/tmp/xeno-canto/Hippolais").expanduser()
    )
    print(red("pipeline.run"))
    output = pipeline.run(dataset)
    print(red("pipeline.get_metrics"))
    metrics = pipeline.get_metrics(dataset, output)
    print(metrics)
