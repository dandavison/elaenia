<p align="center">
    <img width=200px src="https://user-images.githubusercontent.com/52205/74559386-e3a95400-4f29-11ea-9062-57c926547ab7.png" alt="Mountain Elaenia" />
</p>

Elaenia is a collection of transfer learning experiments for identifying bird
species in audio recordings. In all experiments the classifier is
trained on a lower-dimensional embedding of the input audio data,
computed using a publicly-released model (either
[VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
or [BirdNet](https://github.com/kahst/BirdNET)).

The experiments are implemented using a bare-bones ML training pipeline library called [sylph](https://github.com/dandavison/sylph), which I wrote for this project. Sylph allows a pipeline to be defined, comprising a series of data transformation / feature extraction steps, with a final step to train the classifier.

The following Sylph code defines a pipeline which performs preliminary transformations of the raw audio data, computes the spectrogram, computes the VGGish embeddings, trains a classifier, and computes metrics on the test set:

```python
from sylph.learners.svm import SVMLearner
from sylph.pipeline import Compose
from sylph.pipeline import TrainingPipeline
from sylph.transforms.audio import Audio2Audio16Bit
from sylph.transforms.pca import PCA
from sylph.transforms.vggish import Audio2Spectrogram
from sylph.transforms.vggish import Spectrogram2VGGishEmbeddings


pipeline = TrainingPipeline(
    transform=Compose(
        [
            Audio2Audio16Bit(normalize_amplitude=True),
            Audio2Spectrogram(),
            Spectrogram2VGGishEmbeddings(),
            PCA(whiten=True),
        ]
    ),
    learn=SVMLearner(),
)
output = pipeline.run(dataset)
metrics = pipeline.get_metrics(dataset, output)
```

### Example: Melodious and Icterine Warbler contact zone

Melodious Warbler (_H. polyglotta_) and Icterine Warbler (_H. icterina_) are similar members of the genus _Hippolais_ that come into contact in a narrow zone in western Europe. [Results](https://dandavison.github.io/2020/02/Hippolais.html) of classifying audio samples from [xeno-canto](https://www.xeno-canto.org/) are illustrated below.

<table><tr><td><img width=500px src="https://user-images.githubusercontent.com/52205/134423857-59da5d1c-7abd-4b79-ad72-3ab3c78efcbf.png" alt="image" /><sub><br>Plot symbol type indicates a priori data labels (Melodious Warbler to the south-west, Icterine Warbler to the north-east; labels not used in classification); plot symbol color indicates results of applying the learned classifier. It appears that individuals whose vocalisations are classified ambiguously (pink/red) may be more common closer the zone of contact (extreme western Germany / eastern France, south to Switzerland).</sub></td></tr></table>

### Run the tests

```sh
git clone git@github.com:dandavison/elaenia.git
cd elaenia
make init
source env.sh
make test
```

<sub>Mountain Elaenia (_Elaenia frantzi_) by [Daniel Uribe](https://www.flickr.com/photos/birdingtourscolombia/15234111589).</sub>
