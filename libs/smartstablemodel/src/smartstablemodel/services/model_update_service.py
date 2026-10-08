import os
import requests
import zipfile
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ModelUpdateService:
    def __init__(self, models_dir: str, info_endpoint: str, download_endpoint: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.info_endpoint = info_endpoint
        self.download_endpoint = download_endpoint
        self.version_file = self.models_dir / "model_version.json"

    def get_local_version(self) -> Dict[str, Any]:
        """Read local version info from JSON file."""
        if self.version_file.exists():
            try:
                with open(self.version_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read version file: {e}")
        return {}

    def save_local_version(self, version_info: Dict[str, Any]):
        """Save current model info to JSON file."""
        try:
            with open(self.version_file, "w") as f:
                json.dump(version_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version file: {e}")

    def check_and_update(self) -> bool:
        """
        Checks for a new version of the champion model and updates if necessary.
        Returns True if an update was performed, False otherwise.
        """
        try:
            logger.info(f"Checking for model updates from {self.info_endpoint}")
            response = requests.get(self.info_endpoint, timeout=10)
            response.raise_for_status()
            champion_info = response.json()

            local_info = self.get_local_version()

            # Compare versions or run_ids
            # We treat version or run_id mismatch as a need to update
            if champion_info.get("version") != local_info.get(
                "version"
            ) or champion_info.get("run_id") != local_info.get("run_id"):
                logger.info(
                    f"New model version found: {champion_info.get('version')} (Run ID: {champion_info.get('run_id')})"
                )
                self.download_and_extract()
                self.save_local_version(champion_info)
                logger.info("Model updated successfully.")
                return True
            else:
                logger.info(
                    f"Local model is up to date (Version: {local_info.get('version')})."
                )
                return False
        except Exception as e:
            logger.error(f"Failed to check or update model: {e}")
            return False

    def download_and_extract(self):
        """Download model zip and extract it to the models directory."""
        logger.info(f"Downloading model from {self.download_endpoint}")
        response = requests.get(self.download_endpoint, stream=True, timeout=30)
        response.raise_for_status()

        zip_path = self.models_dir / "champion_model.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Extracting model to {self.models_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # We extract directly into models_dir
            zip_ref.extractall(self.models_dir)

        # Cleanup
        if zip_path.exists():
            os.remove(zip_path)

    def find_latest_model(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Finds the latest .tflite model and its corresponding label mapping in the models directory.
        Returns (tflite_path, label_mapping_path).
        """
        tflite_files = list(self.models_dir.glob("*.tflite"))
        if not tflite_files:
            return None, None

        # Sort by mtime to get the latest
        latest_tflite = max(tflite_files, key=os.path.getmtime)

        # Look for label mapping: either {stem}_label_mapping.json or {stem}.json
        label_mapping = self.models_dir / f"{latest_tflite.stem}_label_mapping.json"
        if not label_mapping.exists():
            label_mapping = self.models_dir / f"{latest_tflite.stem}.json"

        if not label_mapping.exists():
            # Fallback to any json if only one exists other than version.json
            json_files = [
                f
                for f in self.models_dir.glob("*.json")
                if f.name != "model_version.json"
            ]
            if len(json_files) == 1:
                label_mapping = json_files[0]
            else:
                label_mapping = None

        return latest_tflite, label_mapping
