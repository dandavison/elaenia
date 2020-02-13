import numpy as np
import tensorflow.compat.v1 as tf

from vendor.tensorflow_models.research.audioset.vggish import vggish_input
from vendor.tensorflow_models.research.audioset.vggish import vggish_params
from vendor.tensorflow_models.research.audioset.vggish import vggish_slim

from elaenia.dataset import Dataset
from elaenia.pipeline import Transform
from elaenia.vggish import get_audio_frames


tf.disable_v2_behavior()


class Audio2Windows(Transform):
    def __call__(self, dataset: Dataset) -> Dataset:
        observations = np.array([get_audio_frames(audio) for audio in dataset.observations])
        return Dataset(observations=observations, labels=dataset.labels)


class Audio2Spectrogram(Transform):
    # If these spectrograms are going to be passed to VGGish, then VGGish features_tensor needs to
    # be a 4D tensor: (n_audio, n_examples_per_audio, n_stft_windows_per_example,
    # n_frequency_bands_per_stft_window) Why though? Why are the examples from different audios not
    # all concatenated in a single dimension? I think this is because pytorch always demands that
    # data is batched. See define_vggish_slim docstring.
    #
    def __call__(self, dataset: Dataset) -> Dataset:
        spectrograms = [
            vggish_input.waveform_to_examples(audio.time_series, audio.sampling_rate)
            for audio in dataset.observations
        ]
        n_examples_per_audio = [s.shape[0] for s in spectrograms]
        observations = np.concatenate(spectrograms, axis=0)
        labels = np.repeat(dataset.labels, n_examples_per_audio)
        dataset = Dataset(observations=observations, labels=labels)
        dataset.n_examples_per_audio = n_examples_per_audio  # Hack
        return dataset


class Spectrogram2VGGishEmbeddings(Transform):
    def __call__(self, dataset: Dataset) -> Dataset:
        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(
                sess, "elaenia/pipelines/vggish/vggish_model.ckpt"
            )
            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run(
                [embedding_tensor], feed_dict={features_tensor: dataset.observations}
            )
        # Hack
        n_examples_per_audio = dataset.n_examples_per_audio
        dataset = Dataset(observations=embedding_batch, labels=dataset.labels)
        dataset.n_examples_per_audio = n_examples_per_audio
        return dataset
