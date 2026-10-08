import logging
import numpy as np
from typing import Dict, List, cast, Any

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

logger = logging.getLogger("audio_evaluation")


def evaluate_segments_by_label(
    self, segments_by_label: Dict[str, List[np.ndarray]], batch_size: int = 32
) -> dict:
    """
    Evaluate classifier on a dict of label -> list[np.ndarray segments].
    Returns per-class precision/recall/F1/support, overall accuracy and confusion matrix.
    """
    # Require label mapping
    if not hasattr(self, "label_to_idx") or not self.label_to_idx:
        return {
            "error": "Classifier has no label mapping. Train the classifier first.",
            "model_version": self.get_model_version(),
        }

    # Flatten dataset
    X: List[np.ndarray] = []
    y_true_labels: List[str] = []
    for label, segs in segments_by_label.items():
        if not isinstance(segs, list) or len(segs) == 0:
            continue
        # Only include labels the classifier knows
        if label not in self.label_to_idx:
            logger.warning(f"Skipping unknown label for model mapping: {label}")
            continue

        for seg in segs:
            if isinstance(seg, tuple):
                X.append(seg[0])
            else:
                X.append(seg)
            y_true_labels.append(label)

    if not X:
        return {
            "error": "No evaluable segments after filtering by known labels.",
            "model_version": self.get_model_version(),
        }

    # Inference: get probability dicts to be robust to binary/multiclass
    prob_dicts: List[Dict[str, float]] = cast(
        List[Dict[str, float]],
        self.infer_audio_segment(
            np.stack(X), batch_size=batch_size, return_probabilities=True
        ),
    )

    # Convert y_true and predictions to indices
    label_to_idx = self.label_to_idx
    idx_to_label = (
        self.idx_to_label
        if hasattr(self, "idx_to_label")
        else {v: k for k, v in label_to_idx.items()}
    )

    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    pred_confidences: List[float] = []

    for true_label, p in zip(y_true_labels, prob_dicts):
        # Predicted label and its confidence
        if not p:
            # Empty probability dict; skip sample
            continue
        pred_label, pred_conf = max(p.items(), key=lambda kv: kv[1])
        if pred_label not in label_to_idx or true_label not in label_to_idx:
            # Skip unknown labels
            continue

        y_true_idx.append(label_to_idx[true_label])
        y_pred_idx.append(label_to_idx[pred_label])
        pred_confidences.append(float(pred_conf))

    if not y_true_idx:
        return {
            "error": "No evaluable samples after mapping to indices.",
            "model_version": self.get_model_version(),
        }

    y_true = np.array(y_true_idx, dtype=int)
    y_pred = np.array(y_pred_idx, dtype=int)

    # Determine classes present in the data (order by index)
    classes_in_data = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    class_labels = [idx_to_label[i] for i in classes_in_data]

    # Metrics
    overall_acc = float(accuracy_score(y_true, y_pred))
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=classes_in_data, zero_division=0
    )
    # Ensure array types for indexing (handles single-class edge cases)
    precisions = np.atleast_1d(precisions)
    recalls = np.atleast_1d(recalls)
    f1_scores = np.atleast_1d(f1_scores)
    supports = np.atleast_1d(cast(np.ndarray, supports))

    conf_mat = confusion_matrix(y_true, y_pred, labels=classes_in_data)

    # Per-class accuracy (TP/support)
    per_class = {}
    # Confidence summaries per predicted class
    correct_conf_sum = {i: 0.0 for i in classes_in_data}
    correct_conf_cnt = {i: 0 for i in classes_in_data}
    incorrect_conf_sum = {i: 0.0 for i in classes_in_data}
    incorrect_conf_cnt = {i: 0 for i in classes_in_data}

    for t, p, conf in zip(y_true, y_pred, pred_confidences):
        if t == p:
            correct_conf_sum[p] += conf
            correct_conf_cnt[p] += 1
        else:
            incorrect_conf_sum[p] += conf
            incorrect_conf_cnt[p] += 1

    # Map class id -> position in confusion matrix
    idx_pos_map = {cls_id: pos for pos, cls_id in enumerate(classes_in_data)}

    # Build per-class dict
    for cls_id, lbl in zip(classes_in_data, class_labels):
        pos = idx_pos_map[cls_id]
        tp = int(conf_mat[pos, pos]) if conf_mat.size > 0 else 0
        sup = int(supports[pos]) if pos < len(supports) else 0
        cls_acc = float(tp / sup) if sup > 0 else 0.0

        avg_conf_correct = (
            float(correct_conf_sum[cls_id] / correct_conf_cnt[cls_id])
            if correct_conf_cnt[cls_id] > 0
            else None
        )
        avg_conf_incorrect = (
            float(incorrect_conf_sum[cls_id] / incorrect_conf_cnt[cls_id])
            if incorrect_conf_cnt[cls_id] > 0
            else None
        )

        per_class[lbl] = {
            "support": sup,
            "precision": float(precisions[pos]) if pos < len(precisions) else 0.0,
            "recall": float(recalls[pos]) if pos < len(recalls) else 0.0,
            "f1": float(f1_scores[pos]) if pos < len(f1_scores) else 0.0,
            "accuracy": cls_acc,
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_incorrect": avg_conf_incorrect,
        }

    # Macro/weighted F1
    macro_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
    total = int(np.sum(supports))
    weighted_f1 = (
        float(np.sum(f1_scores * (supports / max(total, 1)))) if total > 0 else 0.0
    )

    return {
        "model_version": self.get_model_version(),
        "overall": {
            "samples": int(len(y_true)),
            "accuracy": overall_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "num_classes": int(len(classes_in_data)),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": class_labels, "matrix": conf_mat.tolist()},
    }


def calculate_segment_based_metrics(
    ground_truth_segments: List[List[str]], prediction_segments: List[List[str]]
) -> Dict[str, Any]:
    """
    Calculate SED metrics (F-score, Error Rate) at segment level.
    Each input is a list of lists (multiple concurrent labels possible per segment).
    Mesaros et al. (2016) metrics.

    Args:
        ground_truth_segments: List of ground truth label sets for each time segment.
        prediction_segments: List of predicted label sets for each time segment.

    Returns:
        Dict with f_score, error_rate, S, D, I metrics.
    """
    N = 0
    TP, FP, FN = 0, 0, 0
    subs, dels, ins = 0, 0, 0

    for gt, pred in zip(ground_truth_segments, prediction_segments):
        gt_set = set(gt) - {
            "normal",
            "silence",
        }  # Exclude non-event labels from SED metrics
        pred_set = set(pred) - {"normal", "silence"}

        n_gt = len(gt_set)

        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        TP += tp
        FP += fp
        FN += fn
        N += n_gt

        # SED substitutions, deletions, insertions
        s = min(fp, fn)
        d = max(0, fn - fp)
        i = max(0, fp - fn)

        subs += s
        dels += d
        ins += i

    f_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    error_rate = (subs + dels + ins) / N if N > 0 else 0.0

    return {
        "f_score": float(f_score),
        "error_rate": float(error_rate),
        "substitutions": int(subs),
        "deletions": int(dels),
        "insertions": int(ins),
        "total_gt_events": int(N),
        "tp": int(TP),
        "fp": int(FP),
        "fn": int(FN),
    }
