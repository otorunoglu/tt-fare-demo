import os
import pandas as pd

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("HumanSpeechDataset")

# Resolve default data directory relative to this file's location
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "people")


class HumanSpeechDataset:
    def __init__(self, data_dir=None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            # Try multiple possible locations for the data directory
            possible_dirs = [
                Path(__file__).parent / "data" / "people",  # Package relative
                Path(__file__).parent.parent.parent
                / "data"
                / "people",  # Project root relative (if in src/...)
                Path.cwd()
                / "data"
                / "smartstable"
                / "people",  # Standard backend layout
                Path.cwd() / "data" / "people",  # Current working directory relative
            ]

            self.data_dir = None
            for d in possible_dirs:
                if d.exists():
                    self.data_dir = d
                    logger.info(f"Found human speech data at: {self.data_dir}")
                    break

            if self.data_dir is None:
                # Use default relative path but it will fail exists() check below with better error
                self.data_dir = Path(__file__).parent / "data" / "people"

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found. Tried multiple locations including: {self.data_dir}. "
                "Ensure Common Voice data is located in 'data/people' at the project root."
            )

    def load_common_voice_dataset(
        self, language="en", max_samples=1000, min_duration=1.0, max_duration=5.0
    ):
        """
        Load Common Voice dataset from your local files.

        Args:
            language: "en" for English, "fi" for Finnish
            max_samples: Maximum number of audio samples to process
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
        """
        logger.info(f"Loading Common Voice dataset for language: {language}")

        # Find the dataset directory
        cv_dirs = list(self.data_dir.glob(f"cv-corpus-*/{language}"))
        if not cv_dirs:
            raise FileNotFoundError(
                f"No Common Voice dataset found for language '{language}' in {self.data_dir}"
            )

        cv_dir = cv_dirs[0]  # Use the first found directory
        logger.info(f"Using dataset directory: {cv_dir}")

        # Load the validated.tsv file
        tsv_file = cv_dir / "validated.tsv"
        if not tsv_file.exists():
            raise FileNotFoundError(f"validated.tsv not found in {cv_dir}")

        logger.info(f"Reading metadata from: {tsv_file}")
        df = pd.read_csv(tsv_file, sep="\t")
        logger.info(f"Found {len(df)} entries in validated.tsv")

        # Prepare clips directory path
        clips_dir = cv_dir / "clips"
        if not clips_dir.exists():
            logger.warning(f"Clips directory not found: {clips_dir}")
            # Try alternative path structures
            clips_dir = cv_dir

        return self.process_audio_files(
            df, clips_dir, language, max_samples, min_duration, max_duration
        )

    def process_audio_files(
        self,
        df,
        clips_dir,
        language,
        max_samples=1000,
        min_duration=1.0,
        max_duration=5.0,
    ):
        """Process audio files and create speech segments."""
        speech_segments = []
        processed_count = 0

        # Filter for quality recordings (more upvotes than downvotes)
        df_filtered = df[df["up_votes"] >= df["down_votes"]].copy()
        logger.info(f"Filtered to {len(df_filtered)} quality recordings")

        # Shuffle and limit
        df_sample = df_filtered.sample(
            n=min(len(df_filtered), max_samples * 3), random_state=42
        )

        for idx, row in df_sample.iterrows():
            if processed_count >= max_samples:
                break

            audio_path = clips_dir / row["path"]

            # Check if file exists
            if not audio_path.exists():
                continue

            try:
                # Load audio file
                import audio_preprocessing

                audio, sr = audio_preprocessing.load_audio(str(audio_path), 16000)

                # Force mono conversion if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                duration = len(audio) / sr

                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    continue

                # Create 2-second segments to match your training format
                segment_length = 2 * sr  # 2 seconds at 16kHz

                if len(audio) >= segment_length:
                    # Take the first 2 seconds if audio is longer
                    segment = audio[:segment_length]
                else:
                    # Pad if audio is shorter than 2 seconds
                    segment = np.pad(
                        audio, (0, segment_length - len(audio)), mode="constant"
                    )

                segment_id = f"cv#{language}#{row['path']}"
                speech_segments.append((segment.astype(np.float32), segment_id))
                processed_count += 1

                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count} audio files...")

            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {str(e)}")
                continue

        logger.info(f"Successfully processed {len(speech_segments)} speech segments")
        return speech_segments

    def get_mixed_language_dataset(self, max_samples_per_language=500):
        """
        Get a mixed dataset with both English and Finnish speech.

        Args:
            max_samples_per_language: Maximum samples per language

        Returns:
            List of (audio, id) tuples
        """
        all_segments = []

        # Load English samples
        try:
            en_segments = self.load_common_voice_dataset(
                language="en", max_samples=max_samples_per_language
            )
            all_segments.extend(en_segments)
            logger.info(f"Added {len(en_segments)} English speech segments")
        except Exception as e:
            logger.warning(f"Failed to load English dataset: {str(e)}")

        # Load Finnish samples
        try:
            fi_segments = self.load_common_voice_dataset(
                language="fi", max_samples=max_samples_per_language
            )
            all_segments.extend(fi_segments)
            logger.info(f"Added {len(fi_segments)} Finnish speech segments")
        except Exception as e:
            logger.warning(f"Failed to load Finnish dataset: {str(e)}")

        # Shuffle the combined dataset
        import random

        random.shuffle(all_segments)

        logger.info(
            f"Created mixed language dataset with {len(all_segments)} total segments"
        )
        return all_segments

    def prepare_speech_segments(self, language="mixed", max_samples=1000):
        """
        Main method to prepare speech segments for training.

        Args:
            language: "en", "fi", or "mixed"
            max_samples: Total number of samples to return

        Returns:
            List of (audio, id) tuples
        """
        if language == "mixed":
            return self.get_mixed_language_dataset(
                max_samples_per_language=max_samples // 2
            )
        else:
            return self.load_common_voice_dataset(
                language=language, max_samples=max_samples
            )

    def get_evaluation_paths(self, language="en", max_samples=20):
        """
        Quickly grab raw file paths for evaluation injection (GDPR handling)
        """
        logger.info(f"Loading raw evaluation paths for {language}")
        cv_dirs = list(self.data_dir.glob(f"cv-corpus-*/{language}"))
        if not cv_dirs:
            # Silently return empty if not found so test pipelines don't crash
            return []

        cv_dir = cv_dirs[0]
        tsv_file = cv_dir / "validated.tsv"
        if not tsv_file.exists():
            return []

        df = pd.read_csv(tsv_file, sep="\t")
        df_filtered = df[df["up_votes"] >= df["down_votes"]].copy()

        # Seed 42 guarantees we evaluate on the same sample batch for consistency
        df_sample = df_filtered.sample(
            n=min(len(df_filtered), max_samples * 2), random_state=42
        )

        clips_dir = cv_dir / "clips"
        if not clips_dir.exists():
            clips_dir = cv_dir

        paths = []
        for _, row in df_sample.iterrows():
            audio_path = clips_dir / row["path"]
            if audio_path.exists():
                paths.append(str(audio_path))
                if len(paths) >= max_samples:
                    break

        return paths
