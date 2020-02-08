from elaenia.recording import Recording


class ExperimentRecording(Recording):
    @classmethod
    def from_file(cls, path, index, dataset=None, results=None):
        self = super().from_file(path)
        self.index = index
        self.dataset = dataset
        self.results = results
        self.frames_predicted_integer_labels = None
        self.predicted_integer_label = None
        return self

    @property
    def label(self):
        label, file_name = self.id.split("/")
        return label

    @property
    def integer_label(self):
        # TODO: this might not match the SATL code.
        return sorted(self.dataset.label_set, reverse=True).index(self.label)

    @property
    def id(self):
        return str(self.audio_file.relative_to(self.audio_file.parents[1]))

    @property
    def frame_vggish_embeddings(self):
        return self.dataset.frame_vggish_embeddings[
            self.dataset.frame_recording_ids == self.id
        ]
