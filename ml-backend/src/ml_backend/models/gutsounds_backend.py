from ml_backend.models.base_backend import BaseModelBackend
from gutsoundsmodel.core import GutSoundModel   # Example import

class GutSoundsBackend(BaseModelBackend):
    name = "gutsounds"

    def load(self):
        self.model = GutSoundModel.load_default()

    def predict(self, audio_path, metadata):
        return self.model.predict_segments(audio_path)

    def train(self, tasks, config):
        return self.model.train(tasks)

    def get_labels(self):
        return self.model.labels