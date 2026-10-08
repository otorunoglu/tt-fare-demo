# ml_backend/models/smartstable_backend.py

import os
import logging
from ml_backend.models.base_backend import BaseBackend
from ml_backend.smartstable_core.manager_loader import get_multistall_manager, get_model_version_string, get_model_info
from ml_backend.smartstable_core.predict import smartstable_predict
from ml_backend.smartstable_core.training import smartstable_train
from smartstablemodel.labels import get_all_labels

logger = logging.getLogger("SmartStableBackend")


class SmartStableBackend(BaseBackend):

    def __init__(self):
        self.manager = get_multistall_manager()
        model_info = get_model_info()
        logger.info(f"SmartStableBackend initialized with model: {model_info['model_name']} (source: {model_info['source']})")

    # ------------------------------------------------------------
    # SETUP (Label Studio calls once on backend attach)
    # ------------------------------------------------------------
    def get_setup_response(self) -> dict:
        labels = get_all_labels()
        if "abnormal" not in labels:
            labels.append("abnormal")

        version = get_model_version_string()

        return {
            "model_version": version,
            "labels": labels,
        }


    def handle_predict_request(self, task_id: int, ls):
        smartstable_predict(self.manager, task_id, ls)

    def run_training(self, project_id: int, ls):
        smartstable_train(self.manager, project_id, ls)