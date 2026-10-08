import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


def load_report(path: str) -> Dict[str, Any]:
	with open(path, "r") as f:
		return json.load(f)


def to_per_class_df(report: Dict[str, Any]) -> pd.DataFrame:
	per_class = report["metrics"]["per_class"]
	rows = []
	for label, metrics in per_class.items():
		rows.append(
			{
				"class": label,
				"accuracy": float(metrics.get("accuracy") or 0.0),
				"precision": float(metrics.get("precision") or 0.0),
				"recall": float(metrics.get("recall") or 0.0),
				"f1": float(metrics.get("f1") or 0.0),
				"support": int(metrics.get("support") or 0),
				"avg_confidence_correct": float(metrics.get("avg_confidence_correct") or 0.0),
				"avg_confidence_incorrect": float(metrics.get("avg_confidence_incorrect") or 0.0),
			}
		)
	df = pd.DataFrame(rows)
	# Stable sort by support desc then f1 desc
	df = df.sort_values(["support", "f1"], ascending=[False, False]).reset_index(drop=True)
	return df


def get_confusion_matrix(report: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
	cm_dict = report["metrics"]["confusion_matrix"]
	labels = cm_dict["labels"]
	matrix = np.array(cm_dict["matrix"], dtype=np.int64)
	return matrix, labels


def ensure_dir(path: str) -> None:
	Path(path).mkdir(parents=True, exist_ok=True)


def save_fig(fig: Figure, out_dir: str, filename: str, dpi: int) -> str:
	ensure_dir(out_dir)
	output_path = str(Path(out_dir) / filename)
	fig.tight_layout()
	fig.savefig(output_path, dpi=dpi)
	return output_path


def plot_metrics_heatmap(df: pd.DataFrame, out_dir: str, dpi: int = 150, show: bool = False) -> str:
	metrics_cols = ["accuracy", "precision", "recall", "f1"]
	data = df.set_index("class")[metrics_cols]
	fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.45), 0.45 * len(metrics_cols) + 3))
	sns.heatmap(
		data,
		annot=True,
		fmt=".2f",
		cmap="viridis",
		vmin=0.0,
		vmax=1.0,
		cbar_kws={"label": "score"},
		ax=ax,
	)
	ax.set_title("Per-class Metrics Heatmap")
	ax.set_xlabel("Metric")
	ax.set_ylabel("Class")
	outfile = save_fig(fig, out_dir, "per_class_metrics_heatmap.png", dpi)
	if show:
		plt.show()
	plt.close(fig)
	return outfile


def plot_support_bar(df: pd.DataFrame, out_dir: str, dpi: int = 150, show: bool = False) -> str:
	fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.45), 5))
	sns.barplot(data=df, x="class", y="support", color="#4C78A8", ax=ax)
	ax.set_title("Support per Class")
	ax.set_xlabel("Class")
	ax.set_ylabel("# samples")
	# Rotate and right-align class labels for readability
	ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
	outfile = save_fig(fig, out_dir, "support_per_class.png", dpi)
	if show:
		plt.show()
	plt.close(fig)
	return outfile


def plot_confusion_heatmap(
	cm: np.ndarray,
	labels: List[str],
	out_dir: str,
	dpi: int = 150,
	normalize: str = "row",
	show: bool = False,
) -> Tuple[str, Optional[str]]:
	fig1, ax1 = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.6)))
	data1 = cm.astype(float)
	if normalize == "row":
		row_sums = data1.sum(axis=1, keepdims=True)
		row_sums[row_sums == 0] = 1
		data1 = data1 / row_sums
	sns.heatmap(data1, xticklabels=labels, yticklabels=labels, cmap="magma", vmin=0.0, vmax=1.0 if normalize else None, cbar_kws={"label": "proportion" if normalize else "count"}, ax=ax1)
	ax1.set_xlabel("Predicted")
	ax1.set_ylabel("True")
	ax1.set_title("Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix (counts)")
	outfile1 = save_fig(fig1, out_dir, f"confusion_matrix_{'normalized' if normalize else 'counts'}.png", dpi)
	if show:
		plt.show()
	plt.close(fig1)

	# Also save counts version if normalized requested
	outfile2 = None
	if normalize:
		fig2, ax2 = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.6)))
		sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="magma", cbar_kws={"label": "count"}, ax=ax2)
		ax2.set_xlabel("Predicted")
		ax2.set_ylabel("True")
		ax2.set_title("Confusion Matrix (counts)")
		outfile2 = save_fig(fig2, out_dir, "confusion_matrix_counts.png", dpi)
		if show:
			plt.show()
		plt.close(fig2)
	return outfile1, outfile2


def plot_pr_f1_bubble(df: pd.DataFrame, out_dir: str, dpi: int = 150, show: bool = False) -> str:
	# Convert to numpy arrays (explicit float) for type checker friendliness
	x: np.ndarray = df["recall"].to_numpy(dtype=float)
	y: np.ndarray = df["precision"].to_numpy(dtype=float)
	c: np.ndarray = df["f1"].to_numpy(dtype=float)
	s: np.ndarray = df["support"].to_numpy(dtype=float)
	if s.size == 0:
		s_scaled = np.array([], dtype=float)
	else:
		s_max = float(np.max(s))
		s_scaled = 50.0 + 250.0 * (s / s_max) if s_max > 0 else 50.0 + 0.0 * s

	fig, ax = plt.subplots(figsize=(8, 7))
	sc = ax.scatter(x, y, s=s_scaled, c=c, cmap="viridis", edgecolor="black", alpha=0.8)  # type: ignore[arg-type]
	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	ax.set_title("Precision vs Recall (size=support, color=F1)")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label("F1")
	# Annotate points
	for _, row in df.iterrows():
		ax.annotate(row["class"], (row["recall"], row["precision"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
	outfile = save_fig(fig, out_dir, "precision_recall_f1_bubble.png", dpi)
	if show:
		plt.show()
	plt.close(fig)
	return outfile


def plot_3d_scatter(df: pd.DataFrame, out_dir: str, dpi: int = 150, show: bool = False) -> str:
	fig = plt.figure(figsize=(9, 7))
	ax = fig.add_subplot(111, projection="3d")
	x_vals: np.ndarray = df["precision"].to_numpy(dtype=float)
	y_vals: np.ndarray = df["recall"].to_numpy(dtype=float)
	z_vals: np.ndarray = df["f1"].to_numpy(dtype=float)
	s_vals: np.ndarray = df["support"].to_numpy(dtype=float)
	if s_vals.size == 0:
		s_scaled = np.array([], dtype=float)
	else:
		s_max = float(np.max(s_vals))
		s_scaled = 40.0 + 200.0 * (s_vals / s_max) if s_max > 0 else 40.0 + 0.0 * s_vals
	# Use get_cmap for static attribute safety
	plasma_cmap = cm.get_cmap("plasma")
	colors = plasma_cmap(z_vals)
	ax.scatter(x_vals, y_vals, z_vals, s=s_scaled, c=colors, edgecolor="black", alpha=0.85)  # type: ignore[arg-type]
	for _, row in df.iterrows():  # annotation; low volume expected
		ax.text(float(row["precision"]), float(row["recall"]), float(row["f1"]), str(row["class"]), fontsize=8)
	ax.set_xlabel("Precision")
	ax.set_ylabel("Recall")
	ax.set_zlabel("F1")
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.set_zlim(0.0, 1.0)
	ax.set_title("3D Metrics Map (color=F1, size=support)")
	outfile = save_fig(fig, out_dir, "metrics_3d_map.png", dpi)
	if show:
		plt.show()
	plt.close(fig)
	return outfile


def main() -> int:
	parser = argparse.ArgumentParser(description="Visualize per-class metrics and confusion matrix from a report.json")
	parser.add_argument("--input", required=True, help="Path to report JSON (e.g., tools/model_metrics/report_*.json)")
	parser.add_argument("--out", default="./model_metrics/visualizations", help="Directory to save generated figures")
	parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures")
	parser.add_argument("--show", action="store_true", help="Show figures interactively")
	parser.add_argument("--no-show", dest="show", action="store_false", help=argparse.SUPPRESS)
	parser.set_defaults(show=False)
	args = parser.parse_args()

	report = load_report(args.input)
	out_dir = args.out
	ensure_dir(out_dir)

	df = to_per_class_df(report)
	cm, labels = get_confusion_matrix(report)

	outputs = []
	outputs.append(plot_metrics_heatmap(df, out_dir, dpi=args.dpi, show=args.show))
	outputs.append(plot_support_bar(df, out_dir, dpi=args.dpi, show=args.show))
	out_norm, out_counts = plot_confusion_heatmap(cm, labels, out_dir, dpi=args.dpi, normalize="row", show=args.show)
	outputs.append(out_norm)
	if out_counts:
		outputs.append(out_counts)
	outputs.append(plot_pr_f1_bubble(df, out_dir, dpi=args.dpi, show=args.show))
	outputs.append(plot_3d_scatter(df, out_dir, dpi=args.dpi, show=args.show))

	print("Saved:")
	for p in outputs:
		print(" -", p)
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

## HOW TO RUN:
# EG:
# python src/smartstablemodel/scripts/model_metrics/report_visualizer.py \
#   --input scripts/model_metrics/report_25.11.25.json \
#   --out src/smartstablemodel/scripts/model_metrics/visualizations \
#   --dpi 150
# Or: Show interactive windows (e.g., to rotate the 3D plot):
# python src/smartstablemodel/scripts/model_metrics/report_visualizer.py \
#   --input src/smartstablemodel/scripts/model_metrics/report_25.10.09.json \
#   --out src/smartstablemodel/scripts/model_metrics/visualizations \
#   --dpi 150 --show