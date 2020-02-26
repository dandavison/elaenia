import pickle
from pathlib import Path

from sylph.xeno_quero.dataset import XenoQueroDataset

from elaenia.pipelines.birdnet import make_birdnet_embeddings_training_pipeline
from elaenia.utils.color import red


if __name__ == "__main__":
    dataset = XenoQueroDataset.from_species_globs(["Xiphorhynchus-*"], training_proportion=0.8)
    pipeline = make_birdnet_embeddings_training_pipeline(
        birdnet_output_dir=Path("~/tmp/elaenia/birdnet_embeddings_Xiphorhynchus").expanduser()
    )
    print(red("pipeline.run"))
    output = pipeline.run(dataset)
    print(red("pipeline.get_metrics"))
    metrics = pipeline.get_metrics(dataset, output)
    print(metrics)
    with Path("BirdNet-TL-Xiphorhynchus.pickle").open("wb") as fp:
        pickle.dump({"output": output, "metrics": metrics}, fp)
