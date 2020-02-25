from pathlib import Path

from sylph.xeno_quero.dataset import XenoQueroDataset

from elaenia.pipelines.birdnet import make_birdnet_embeddings_training_pipeline
from elaenia.utils.color import red


if __name__ == "__main__":
    dataset = XenoQueroDataset.from_species_globs(["Xiphorhynchus-*"], training_proportion=0.8)
    # birdnet_output_dir=Path("~/tmp/elaenia/birdnet_embeddings_Xiphorhynchus").expanduser()
    pipeline = make_birdnet_embeddings_training_pipeline()
    print(red("pipeline.run"))
    output = pipeline.run(dataset)
    print(red("pipeline.get_metrics"))
    metrics = pipeline.get_metrics(dataset, output)
    print(metrics)
