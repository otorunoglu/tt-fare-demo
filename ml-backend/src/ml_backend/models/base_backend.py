class BaseBackend:
    """Backend interface for ML backends (SmartStable, GutSounds, etc)."""

    def get_setup_response(self) -> dict:
        raise NotImplementedError

    def handle_predict_request(self, task_id: int, ls):
        raise NotImplementedError

    def run_training(self, project_id: int, ls):
        raise NotImplementedError