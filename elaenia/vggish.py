from elaenia.recording import Recording
from vendor.tensorflow_models.research.audioset.vggish import mel_features
from vendor.tensorflow_models.research.audioset.vggish import vggish_input


def get_stft_frames(recording: Recording):
    """
    Return log-mel STFT reshaped into frames.

    Note that what I am calling "frames", tensorflow calls "examples".
    What tensorflow calls "frames" are 10ms time windows within each example.

    Output is (n_frames, frame_length, n_bands) = (n_frames, 96, 64)
    """
    return vggish_input.waveform_to_examples(recording.time_series, recording.sampling_rate)


def get_audio_frames(recording: Recording):
    """
    Return raw audio waveform broken into 0.96 s frames.

    Output is (n_frames, frame_length)
    """
    # This is based on the implementation of vggish_input.waveform_to_examples.
    return mel_features.frame(
        recording.time_series,
        window_length=int(round(0.96 * recording.sampling_rate)),
        hop_length=int(round(0.96 * recording.sampling_rate)),
    )


def get_frame_energies(frames):
    """
    Return sum of squared amplitudes within each frame.
    """
    return (frames ** 2).sum(axis=1)
