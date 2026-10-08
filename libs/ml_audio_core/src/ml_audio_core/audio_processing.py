"""
Generic ML audio processing utility module.
Provides reusable, centralized audio manipulation, feature extraction, and preprocessing functions
for general-purpose machine learning audio workflows.
"""

import hashlib

import numpy as np
import audio_preprocessing
import librosa
import soundfile as sf
from skimage.transform import resize
from scipy.signal import resample_poly
from typing import Dict, List, Tuple, Generator, Optional
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml_audio_core.audio_processing")


class AudioProcessor:
    """
    General-purpose audio processing utility for machine learning workflows.

    Provides reusable operations for audio preprocessing, feature extraction,
    segmentation, and batch preparation for ML models.
    """

    def __init__(
        self,
        sr=16000,
        win_seconds=2.0,
        win_overlap=0.5,
        n_mels=64,
        hop_length=512,
        fixed_tbins=96,
        backend="rust",
    ):

        import warnings

        warnings.warn(
            "AudioProcessor's feature extraction (mel spectrograms) is deprecated. "
            "New models handle preprocessing internally (e.g., via PannsEmbedder). "
            "This class should only be used for file loading and windowing.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.sr = sr
        self.win_seconds = win_seconds
        self.win_overlap = win_overlap
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fixed_tbins = fixed_tbins
        self.backend = backend

        logger.info(
            f"AudioProcessor initialized: SR={sr}, WIN_SECONDS={win_seconds}, Backend={backend}"
        )

    def audio_to_mel_spectrogram(
        self, audio_segment: np.ndarray, target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a normalized mel spectrogram from an audio segment.

        Args:
            audio_segment: Input audio segment as a 1D numpy array.
            target_length: Target length for audio segment (defaults to win_seconds).

        Returns:
            Mel spectrogram array with shape (n_mels, fixed_tbins), no channel dimension.
        """
        # Ensure correct length
        if target_length is None:
            target_length = int(self.sr * self.win_seconds)

        audio_segment = self._prepare_audio_segment(audio_segment, target_length)

        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment.astype(np.float32),
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        )

        # Use PCEN consistently (better for anomaly detection)
        mel_spec = librosa.pcen(mel_spec, sr=self.sr)
        # Normalize to [0, 1] range consistently
        # Check for invalid values before normalization
        if np.any(np.isnan(mel_spec)) or np.any(np.isinf(mel_spec)):
            logger.warning(
                "Found NaN or Inf values in mel spectrogram, replacing with zeros"
            )
            mel_spec = np.nan_to_num(mel_spec, nan=0.0, posinf=1.0, neginf=0.0)

        # More robust normalization
        mel_min = np.min(mel_spec)
        mel_max = np.max(mel_spec)
        mel_range = mel_max - mel_min

        if mel_range > 1e-8:  # Avoid division by zero
            mel_spec = (mel_spec - mel_min) / mel_range
        else:
            logger.warning("Mel spectrogram has no dynamic range, setting to zeros")
            mel_spec = np.zeros_like(mel_spec)

        # Ensure values are in [0, 1] range
        mel_spec = np.clip(mel_spec, 0.0, 1.0)

        # Resize to fixed time bins if needed
        if mel_spec.shape[1] != self.fixed_tbins:
            mel_spec = resize(
                mel_spec, (self.n_mels, self.fixed_tbins), anti_aliasing=True
            )

        # Final validation
        if np.any(np.isnan(mel_spec)) or np.any(np.isinf(mel_spec)):
            logger.error("Still have NaN/Inf after processing!")
            mel_spec = np.zeros((self.n_mels, self.fixed_tbins))

        # Log statistics for debugging (only occasionally to avoid spam)
        if np.random.random() < 0.01:  # 1% of the time
            logger.debug(
                f"Mel stats: min={np.min(mel_spec):.6f}, max={np.max(mel_spec):.6f}, mean={np.mean(mel_spec):.6f}"
            )

        return mel_spec.astype(np.float32)

    def prepare_mel_batch(self, mel_list: List[np.ndarray]) -> np.ndarray:
        """
        Prepare a batch of mel spectrograms for ML model input.

        Args:
            mel_list: List of mel spectrograms from audio_to_mel_spectrogram.

        Returns:
            4D array with shape (batch, n_mels, fixed_tbins, 1).
        """
        if not mel_list:
            raise ValueError("Empty mel_list provided")

        # Stack and add channel dimension
        mels_arr = np.stack(mel_list)  # (batch, n_mels, fixed_tbins)

        # Ensure 4D shape: (batch, n_mels, fixed_tbins, 1)
        if len(mels_arr.shape) == 3:
            mels_arr = np.expand_dims(mels_arr, axis=-1)
        elif len(mels_arr.shape) == 4 and mels_arr.shape[-1] != 1:
            logger.warning(
                f"Unexpected channel dimension: {mels_arr.shape[-1]}, taking first channel"
            )
            mels_arr = mels_arr[..., :1]

        return mels_arr.astype(np.float32)

    def _prepare_audio_segment(
        self, audio_segment: np.ndarray, target_length: int
    ) -> np.ndarray:
        """Ensure audio segment is mono, float32, and has the correct length."""
        # Convert to mono if stereo
        if len(audio_segment.shape) > 1:
            audio_segment = np.mean(audio_segment, axis=1)

        # Adjust length
        if len(audio_segment) < target_length:
            # Pad with zeros
            audio_segment = np.pad(
                audio_segment, (0, target_length - len(audio_segment))
            )
        elif len(audio_segment) > target_length:
            # Truncate
            audio_segment = audio_segment[:target_length]

        return audio_segment.astype(np.float32)

    def load_and_resample_audio(
        self, file_path: str, offset: float = 0.0, duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and resample it to the target sample rate using Rust backend.

        Args:
            file_path: Path to audio file.
            offset: Start reading at this time (in seconds).
            duration: Only load up to this much audio (in seconds).

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        try:
            if self.backend == "rust":
                # Load full audio via Rust
                audio, sr = audio_preprocessing.load_audio(str(file_path), self.sr)
            else:
                # Load via Librosa (legacy/slower)
                # librosa.load supports resampling directly if sr is provided
                audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
                # Ensure float32
                audio = audio.astype(np.float32)

            # Handle offset and duration via slicing (since Rust decoder loads full file currently)
            # Librosa load also loads full file by default unless offset/duration passed but user might want consistent slicing logic
            # However, prompt implies we want to switch the implementation.
            # If using librosa, we can use its offset/duration params for efficiency:
            if self.backend != "rust" and (offset > 0 or duration is not None):
                # Reload with offset/duration if we didn't use them (simpler to just load normally then slice or re-load?)
                # To keep it simple and consistent with the Rust path logic (which slices after load),
                # let's just use the slicing logic below for both unless performance is critical.
                # Actually, librosa.load takes offset/duration.
                pass

            # Simpler: Just rely on common slicing logic below for now to ensure exact parity
            # unless we change librosa call above.
            # Let's adjust librosa call to use params if possible?
            # Creating consistent slicing logic is safer for parity.

            if offset > 0 or duration is not None:
                start_sample = int(offset * sr)
                end_sample = len(audio)
                if duration is not None:
                    end_sample = min(len(audio), start_sample + int(duration * sr))

                if start_sample < len(audio):
                    audio = audio[start_sample:end_sample]
                else:
                    audio = np.array([], dtype=np.float32)

            return audio.astype(np.float32), sr
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file {file_path} with backend {self.backend}: {e}"
            )
        
    def blob_hash(self, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)  # defensive: lock dtype + order
        tag = f"sr={self.sr};ch=1;dt=f32".encode()
        return hashlib.sha256(tag + audio.tobytes()).hexdigest()[:16]

    def extract_blob(self, source_file, start, end):
        audio, sr = self.load_and_resample_audio(
            source_file, offset=start, duration=end - start
        )
        return self.blob_hash(audio), audio, sr
    
    def load_full(self, source_file):
        """Resample the whole file once. The expensive call — do it per task, not per segment."""
        audio, sr = self.load_and_resample_audio(source_file)   # no offset/duration -> full file
        return np.ascontiguousarray(audio, dtype=np.float32), sr

    def slice_and_hash(self, full_audio, sr, start, end):
        """Slice a segment from already-resampled audio. Must match the old per-segment
        slicing byte-for-byte so hashes stay stable."""
        start_sample = int(start * sr)
        end_sample = min(len(full_audio), start_sample + int((end - start) * sr))
        if start_sample < len(full_audio):
            seg = full_audio[start_sample:end_sample]
        else:
            seg = np.array([], dtype=np.float32)
        seg = np.ascontiguousarray(seg, dtype=np.float32)
        return self.blob_hash(seg), seg

    def create_sliding_windows(
        self,
        audio: np.ndarray,
        segment_duration: Optional[float] = None,
        overlap: Optional[float] = None,
        source_name: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[str]]:
        """
        Create overlapping sliding windows from an audio array.

        Args:
            audio: Input audio array.
            segment_duration: Duration of each segment (defaults to win_seconds).
            overlap: Overlap fraction between segments (defaults to win_overlap).
            source_name: Optional name of the audio source (e.g. filename) for ID generation.

        Returns:
            Tuple of (window_list, time_list, id_list), where time_list contains (start, end) tuples.
        """
        segment_duration = segment_duration or self.win_seconds
        overlap = overlap or self.win_overlap

        segment_samples = int(segment_duration * self.sr)
        hop_samples = int(segment_samples * (1 - overlap))

        windows = []
        times = []
        ids = []

        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            window = audio[start : start + segment_samples]
            start_time = start / self.sr
            end_time = (start + segment_samples) / self.sr

            windows.append(window)
            times.append((start_time, end_time))

            # Generate ID: "source#timestamp" or just "timestamp" if source unknown
            prefix = source_name if source_name else "audio"
            ids.append(f"{prefix}#{start_time:.3f}")

        return windows, times, ids

    def load_varying_length_audio(
        self,
        file_path: str,
        duration: Optional[float] = None,
        overlap: Optional[float] = None,
        include_metadata: bool = False,
    ) -> List[np.ndarray] | Tuple[List[np.ndarray], List[str]]:
        """
        Load an audio file and split it into overlapping segments.

        Args:
            file_path: Path to audio file.
            duration: Duration of each segment.
            overlap: Overlap fraction between segments.
            include_metadata: If True, returns (segments, ids).

        Returns:
            List of audio segments, or (segments, ids) if include_metadata is True.
        """
        duration = duration or self.win_seconds
        overlap = overlap or self.win_overlap

        # Load and resample audio
        audio, _ = self.load_and_resample_audio(file_path)

        # Create sliding windows
        import os

        source_name = os.path.basename(file_path)
        windows, _, ids = self.create_sliding_windows(
            audio, duration, overlap, source_name=source_name
        )

        if include_metadata:
            return windows, ids
        return windows

    def load_varying_length_audio_to_melspectrogram(
        self,
        file_path: str,
        duration: Optional[float] = None,
        overlap: Optional[float] = None,
        use_pcen: bool = True,
    ) -> np.ndarray:
        """
        Load an audio file and convert it to a batch of mel spectrograms.

        Args:
            file_path: Path to audio file.
            duration: Duration of each segment.
            overlap: Overlap fraction between segments.
            use_pcen: Whether to use PCEN normalization.

        Returns:
            Array of mel spectrograms with shape (n_segments, n_mels, fixed_tbins, 1).
        """
        # Load audio segments
        segments = self.load_varying_length_audio(file_path, duration, overlap)

        # Convert to mel spectrograms
        mel_specs = []
        for segment in segments:
            mel_spec = self.audio_to_mel_spectrogram(segment)
            mel_specs.append(mel_spec[..., np.newaxis])  # Add channel dimension

        return np.array(mel_specs)

    def load_audio_files_from_folder_to_mel(
        self, folder_path: str, duration: Optional[float] = None, use_pcen: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load all audio files from a folder and convert them to mel spectrograms.

        Args:
            folder_path: Path to folder containing audio files.
            duration: Duration for each file segment (if None, uses entire file).
            use_pcen: Whether to use PCEN normalization.

        Returns:
            Tuple of (mel_spectrogram_array, file_names).
        """
        import os

        # Get all audio files
        audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
        audio_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.splitext(f.lower())[1] in audio_extensions
        ]

        if not audio_files:
            raise ValueError(f"No audio files found in {folder_path}")

        mel_specs = []
        file_names = []

        for file_name in sorted(audio_files):
            try:
                file_path = os.path.join(folder_path, file_name)

                if duration is None:
                    # Load entire file as single segment
                    audio, _ = self.load_and_resample_audio(file_path)
                    mel_spec = self.audio_to_mel_spectrogram(audio)
                    mel_specs.append(mel_spec[..., np.newaxis])
                    file_names.append(file_name)
                else:
                    # Load file with segmentation
                    file_mel_specs = self.load_varying_length_audio_to_melspectrogram(
                        file_path, duration=duration, use_pcen=use_pcen
                    )
                    mel_specs.extend(file_mel_specs)
                    file_names.extend(
                        [f"{file_name}_seg_{i}" for i in range(len(file_mel_specs))]
                    )

            except Exception as e:
                logger.warning(f"Failed to process {file_name}: {str(e)}")

        return np.array(mel_specs), file_names

    def load_audio_streaming(
        self, file_path: str, chunk_size: int = 1024 * 1024
    ) -> Generator[np.ndarray, None, None]:
        """Generator for streaming audio file loading (memory efficient for large files)."""
        with sf.SoundFile(file_path) as f:
            while True:
                chunk = f.read(chunk_size)
                if len(chunk) == 0:
                    break

                # Convert to mono if stereo
                if len(chunk.shape) > 1:
                    chunk = np.mean(chunk, axis=1)

                # Resample if necessary
                if f.samplerate != self.sr:
                    chunk = audio_preprocessing.resample_numpy(
                        chunk, orig_sr=f.samplerate, target_sr=self.sr
                    )

                yield chunk.astype(np.float32)

    def prepare_for_pann(self, audio_segment: np.ndarray) -> np.ndarray:
        """Prepare audio segment for input to PANN (or similar ML model) inference."""
        audio_segment = self._prepare_audio_segment(
            audio_segment, int(self.sr * self.win_seconds)
        )
        return np.expand_dims(audio_segment, 0)  # Add batch dimension

    def extract_segments_from_ls_annotations_multiclass(
        self,
        audio_path: str,
        annotations: List[dict],
        accepted_labels: List[str],
        min_overlap: float = 0.5,
        min_coverage: float = 0.7,
        include_metadata: bool = False,
    ) -> Dict[str, List[np.ndarray]] | Dict[str, List[Tuple[np.ndarray, str]]]:
        """
        Extract labeled audio segments for multi-class training from time-region annotations.

        Args:
            audio_path: Path to the audio file.
            annotations: List of annotation results, each with time regions and class labels.
            min_overlap: Minimum fraction of segment that must overlap a label region (for long events).
            min_coverage: Minimum fraction of the label region that must be covered by the segment (for short events).
            accepted_labels: Only these labels will be considered.
            include_metadata: If True, returns segments as (audio, id) tuples.

        Returns:
            Dict[label_value] -> List[np.ndarray segments] or List[(audio, id)]
        """
        from typing import Any

        # Load full audio split into fixed windows for training
        result = self.load_varying_length_audio(audio_path, include_metadata=True)
        full_audio_segments, segment_ids = result

        # Build label regions from annotations
        regions_by_label: Dict[str, List[Tuple[float, float]]] = {}
        for result_item in annotations or []:
            val = result_item.get("value", {})
            labels = val.get("labels", []) or []
            start_time = val.get("start")
            end_time = val.get("end")
            if start_time is None or end_time is None:
                logger.warning(
                    f"Skipping annotation with missing start/end: {result_item}"
                )
                continue

            for raw_label in labels:
                if accepted_labels and raw_label not in accepted_labels:
                    logger.info(
                        f"Skipping label '{raw_label}' not in accepted labels. Accepted labels: {accepted_labels}"
                    )
                    continue
                regions_by_label.setdefault(raw_label, []).append(
                    (float(start_time), float(end_time))
                )

        logger.info(
            f"DEBUG: Found {list(regions_by_label.keys())} labels in task annotations"
        )

        # If nothing labeled, return empty
        if not regions_by_label:
            logger.info(f"No labeled regions found in annotations for {audio_path}")
            return {}

        # Assign each fixed window segment to a single label if overlap >= min_overlap and unambiguous
        segments_by_label: Dict[str, List[Any]] = {}
        segment_duration = self.win_seconds
        # Note: step_time logic assumes create_sliding_windows default overlap
        overlap = self.win_overlap
        step_time = segment_duration * (1 - overlap)

        for i, segment in enumerate(full_audio_segments):
            segment_id = segment_ids[i]
            segment_start = i * step_time
            segment_end = segment_start + segment_duration

            # Compute max overlap per label
            best_label = None
            best_score = 0.0  # Score can be overlap or coverage
            best_via_short_coverage = False
            competing = False

            for label, regions in regions_by_label.items():
                # compute maximum overlap with any region for this label
                max_ov = 0.0
                max_coverage = 0.0

                for a, b in regions:
                    event_len = b - a
                    intersection = max(0.0, min(segment_end, b) - max(segment_start, a))
                    max_ov = max(max_ov, intersection)
                    if event_len > 0:
                        max_coverage = max(max_coverage, intersection / event_len)

                frac_overlap = (
                    max_ov / segment_duration if segment_duration > 0 else 0.0
                )

                # Determine if this label is a candidate
                is_candidate = False
                score = 0.0

                # Condition 1: Overlaps significantly with the window (Long events)
                if frac_overlap >= min_overlap:
                    is_candidate = True
                    score = frac_overlap
                    candidate_via_short_coverage = False

                # Condition 2: The window covers most of the event (Short events)
                elif max_coverage >= min_coverage:
                    is_candidate = True
                    score = max_coverage  # Prioritize coverage for short events
                    candidate_via_short_coverage = True

                if is_candidate:
                    if score > best_score:
                        competing = False
                        best_score = score
                        best_label = label
                        best_via_short_coverage = candidate_via_short_coverage
                    elif score == best_score and best_label is not None:
                        # tie - ambiguous
                        competing = True

            # Keep only clear assignments
            if best_label is not None and not competing:
                # We already validated threshold logic above
                if include_metadata:
                    if best_via_short_coverage:
                        segment_id = f"{segment_id}#shortcov"
                    segments_by_label.setdefault(best_label, []).append(
                        (segment, segment_id)
                    )
                else:
                    segments_by_label.setdefault(best_label, []).append(segment)

        # Log counts
        for label, segs in segments_by_label.items():
            logger.info(
                f"Extracted {len(segs)} segments for label '{label}' from {audio_path}"
            )

        return segments_by_label

    def extract_unlabeled_segments(
        self,
        audio_path: str,
        annotated_ranges: List[Tuple[float, float]],
        segment_duration: Optional[float] = None,
        min_gap_duration: Optional[float] = None,
        max_segments: Optional[int] = None,
        shuffle: bool = True,
        include_metadata: bool = False,
    ) -> List[np.ndarray] | List[Tuple[np.ndarray, str]]:
        """
        Extract audio segments from unlabeled (gap) regions of an audio file.

        Useful for generating negative/background class samples for binary classification
        when only positive examples are labeled.

        Args:
            audio_path: Path to the audio file.
            annotated_ranges: List of (start_time, end_time) tuples for labeled regions.
            segment_duration: Duration of each extracted segment (defaults to win_seconds).
            min_gap_duration: Minimum gap duration to extract from (defaults to segment_duration).
            max_segments: Maximum number of segments to return (None = no limit).
            shuffle: Whether to shuffle segments before limiting.
            include_metadata: If True, returns segments as (audio, id) tuples.

        Returns:
            List of audio segment arrays from unlabeled regions, or (audio, id) tuples.
        """
        segment_duration = segment_duration or self.win_seconds
        min_gap_duration = min_gap_duration or segment_duration

        try:
            # Load audio and get duration
            audio, sr = self.load_and_resample_audio(audio_path)
            total_duration = len(audio) / sr

            import os

            source_name = os.path.basename(audio_path)

            # Merge overlapping annotated ranges
            gaps = self._find_unlabeled_gaps(
                annotated_ranges, total_duration, min_gap_duration
            )

            if not gaps:
                logger.debug(f"No unlabeled gaps found in {audio_path}")
                return []

            # Extract non-overlapping segments from gaps
            segments = []
            samples_per_segment = int(segment_duration * sr)

            for gap_start, gap_end in gaps:
                gap_duration = gap_end - gap_start
                num_segments = int(gap_duration / segment_duration)

                for i in range(num_segments):
                    seg_start_time = gap_start + i * segment_duration
                    seg_end_time = seg_start_time + segment_duration

                    if seg_end_time <= gap_end:
                        # Convert time to samples
                        start_sample = int(seg_start_time * sr)
                        end_sample = start_sample + samples_per_segment

                        if end_sample <= len(audio):
                            segment = audio[start_sample:end_sample]
                            # Validate segment length (allow 10% tolerance)
                            if len(segment) >= int(samples_per_segment * 0.9):
                                if include_metadata:
                                    segment_id = f"{source_name}#{seg_start_time:.3f}"
                                    segments.append((segment, segment_id))
                                else:
                                    segments.append(segment)

            logger.debug(
                f"Extracted {len(segments)} unlabeled segments from {audio_path}"
            )

            # Shuffle and limit if requested
            if shuffle and segments:
                indices = np.random.permutation(len(segments))
                segments = [segments[i] for i in indices]

            if max_segments is not None and len(segments) > max_segments:
                segments = segments[:max_segments]

            return segments

        except Exception as e:
            logger.warning(
                f"Failed to extract unlabeled segments from {audio_path}: {e}"
            )
            return []

    def compose_short_event_with_background(
        self,
        short_event: np.ndarray,
        background_pool: List[np.ndarray],
        target_duration_seconds: Optional[float] = None,
        crossfade_ms: int = 40,
        position_mode: str = "random",
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Compose a short event onto a background segment with boundary crossfades.

        Args:
            short_event: Event waveform (expected shorter than target duration).
            background_pool: Candidate background waveforms.
            target_duration_seconds: Output duration in seconds (defaults to win_seconds).
            crossfade_ms: Crossfade duration at insertion boundaries.
            position_mode: Event placement mode: "random" or "center".
            rng: Optional numpy Generator for deterministic sampling.

        Returns:
            Composited waveform of length target_duration_seconds * sr.
        """
        if not background_pool:
            raise ValueError("background_pool must contain at least one segment")

        target_duration_seconds = target_duration_seconds or self.win_seconds
        target_samples = int(target_duration_seconds * self.sr)
        if target_samples <= 0:
            raise ValueError("target_duration_seconds must produce at least one sample")

        if rng is None:
            rng = np.random.default_rng()

        # Pick and normalize one background segment to target length.
        bg_idx = int(rng.integers(0, len(background_pool)))
        background = self._prepare_audio_segment(background_pool[bg_idx], target_samples)

        event = np.asarray(short_event, dtype=np.float32)
        if event.ndim > 1:
            event = np.mean(event, axis=1)
        if len(event) == 0:
            return background.astype(np.float32)

        event_len = min(len(event), target_samples)
        event = event[:event_len].astype(np.float32, copy=False)

        max_start = max(0, target_samples - event_len)
        if position_mode == "center":
            start = max_start // 2
        elif position_mode == "random":
            start = int(rng.integers(0, max_start + 1))
        else:
            raise ValueError(
                f"Unsupported position_mode '{position_mode}'. Use 'random' or 'center'."
            )

        composed = background.copy()
        end = start + event_len
        bg_slice = composed[start:end].copy()

        # Crossfade only the boundaries to avoid hard splice artifacts.
        crossfade_samples = int(self.sr * max(0, crossfade_ms) / 1000)
        crossfade_samples = min(crossfade_samples, event_len // 2)

        if crossfade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
            fade_out = fade_in[::-1]

            blended = event.copy()
            blended[:crossfade_samples] = (
                bg_slice[:crossfade_samples] * (1.0 - fade_in)
                + event[:crossfade_samples] * fade_in
            )
            blended[-crossfade_samples:] = (
                bg_slice[-crossfade_samples:] * (1.0 - fade_out)
                + event[-crossfade_samples:] * fade_out
            )
            composed[start:end] = blended
        else:
            composed[start:end] = event

        return np.clip(composed, -1.0, 1.0).astype(np.float32)

    def _find_unlabeled_gaps(
        self,
        annotated_ranges: List[Tuple[float, float]],
        total_duration: float,
        min_gap_duration: float,
    ) -> List[Tuple[float, float]]:
        """
        Find time gaps that are not covered by any annotated range.

        Args:
            annotated_ranges: List of (start, end) tuples for labeled regions.
            total_duration: Total duration of the audio file.
            min_gap_duration: Minimum gap duration to include.

        Returns:
            List of (start, end) tuples for unlabeled gaps.
        """
        logger.debug(
            f"Finding gaps: total_duration={total_duration:.1f}s, "
            f"min_gap={min_gap_duration:.1f}s, num_ranges={len(annotated_ranges)}"
        )

        if not annotated_ranges:
            # Entire file is unlabeled
            if total_duration >= min_gap_duration:
                logger.debug(
                    f"No annotations - entire file is a gap: (0, {total_duration:.1f})"
                )
                return [(0.0, total_duration)]
            return []

        # Sort and merge overlapping ranges
        sorted_ranges = sorted(annotated_ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]

        for start, end in sorted_ranges[1:]:
            if start <= merged[-1][1]:
                # Overlapping or adjacent - extend
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        logger.debug(
            f"Merged {len(annotated_ranges)} ranges into {len(merged)} non-overlapping ranges"
        )

        # Find gaps between merged ranges
        gaps = []
        prev_end = 0.0

        for start, end in merged:
            if start > prev_end:
                gap_duration = start - prev_end
                if gap_duration >= min_gap_duration:
                    gaps.append((prev_end, start))
                    logger.debug(
                        f"  Gap found: ({prev_end:.1f}, {start:.1f}) = {gap_duration:.1f}s"
                    )
            prev_end = max(prev_end, end)

        # Check for gap at the end
        if prev_end < total_duration:
            gap_duration = total_duration - prev_end
            if gap_duration >= min_gap_duration:
                gaps.append((prev_end, total_duration))
                logger.debug(
                    f"  Gap at end: ({prev_end:.1f}, {total_duration:.1f}) = {gap_duration:.1f}s"
                )

        logger.debug(f"Total gaps found: {len(gaps)}")
        return gaps


# STATIC FUNCTIONS
def extract_loudness_features(audio_segment) -> float:
    """
    Return a scalar loudness value for a single mono float32 audio segment.

    Args:
        audio_segment: 1D np.ndarray (float32) in roughly [-1.0, 1.0].
            Should be produced by AudioProcessor._prepare_audio_segment and sliding windows.

    Returns:
        float in [0,1] (heuristic loudness). 0 ~ silence, 1 ~ near full-scale.
    """
    if audio_segment is None:
        return 0.0
    audio = np.asarray(audio_segment)
    if audio.ndim != 1:
        # Fallback to mono mean if shape unexpected
        audio = audio.mean(axis=-1)
    # Ensure float32
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return 0.0
    # RMS
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    # Convert to dBFS (avoid log(0))
    dbfs = 20.0 * np.log10(rms + 1e-12)
    # Clamp floor
    min_db = -100.0
    if dbfs < min_db:
        dbfs = min_db

    return float(dbfs)


def extract_spectral_features(
    audio_segment,
    sample_rate: int = 16000,
    high_freq_cutoff_hz: float = 4000.0,
) -> tuple[float, float]:
    """
    Return basic spectral features for a single mono float32 audio segment.

    Returns:
        (spectral_centroid_hz, high_freq_ratio)
        - spectral_centroid_hz: energy-weighted average frequency in Hz.
        - high_freq_ratio: energy above cutoff / total energy in [0, 1].
    """
    if audio_segment is None:
        return 0.0, 0.0

    audio = np.asarray(audio_segment)
    if audio.ndim != 1:
        # Fallback to mono mean if shape unexpected
        audio = audio.mean(axis=-1)

    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0 or sample_rate <= 0:
        return 0.0, 0.0

    # RFFT magnitudes are sufficient; use power as energy proxy.
    spectrum = np.fft.rfft(audio)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / float(sample_rate))

    total_energy = float(np.sum(power))
    if total_energy <= 0.0:
        return 0.0, 0.0

    centroid = float(np.sum(freqs * power) / total_energy)
    hf_energy = float(np.sum(power[freqs > float(high_freq_cutoff_hz)]))
    hf_ratio = hf_energy / total_energy

    return centroid, float(np.clip(hf_ratio, 0.0, 1.0))
