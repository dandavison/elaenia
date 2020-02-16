from elaenia.datasets.Xiphorhynchus_guttatus_vs_elegans import (
    Xiphorhynchus_guttatus_vs_elegans_Dataset,
)
from elaenia.pipelines.vggish import vggish_svm_learner_pipeline as pipeline

dataset = Xiphorhynchus_guttatus_vs_elegans_Dataset()
output = pipeline.run(dataset)
metrics = pipeline.get_metrics(dataset, output)
print(metrics)