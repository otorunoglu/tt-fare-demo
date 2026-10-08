from ml_backend.models.smartstable_backend import SmartStableBackend
# from ml_backend.models.gutsounds_backend import GutSoundsBackend


# Project→model mapping
# Replace with actual Label Studio project IDs
PROJECT_BACKENDS = {
    1: SmartStableBackend(),
    # 4: GutSoundsBackend(),
}


def load_backend_for_project(project_id: int):
    if project_id is None:
        return None
    return PROJECT_BACKENDS.get(int(project_id))