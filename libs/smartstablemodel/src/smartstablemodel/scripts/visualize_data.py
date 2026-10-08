"""
Visualize historical data directly from the SQLite database (no decider).

This script reads events from the DB and renders duration heatmaps (by group or label)
for a configurable time window.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import matplotlib.pyplot as plt

from smartstablemodel.config import load_metamodel_config
from smartstablemodel.services.data_store import DataStore
from smartstablemodel.services.visualizer import Visualizer
from smartstablemodel.domain.warning import WarningType


# -------------------- Configuration constants --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = "soundscape.db"

# Window selection (choose one approach)
WINDOW_START_ISO: Optional[str] = "2025-08-24 19:00:00"   # e.g., "2025-08-29 18:00:00" or None for earliest
WINDOW_END_ISO: Optional[str] = "2025-08-25 09:00:00"     # e.g., "2025-08-30 09:00:00" or None for latest
WINDOW_LAST_SECONDS: Optional[int] = None  # e.g., 8 * 3600

# Heatmap and plotting options
BIN_SECONDS = 120
TICK_MINUTES = 15
GROUP_LABELS = False
TOP_N = 20
MIN_TOTAL_SECONDS = 2.0
MAX_BINS = 1000
MAX_CELL_SECONDS: Optional[float] = 30.0
TIME_FORMAT = "%H:%M"

# Multi‑day overview options
MULTI_DAY_ENABLED = False
MULTI_DAY_DAYS = 14
MULTI_DAY_BIN_SECONDS = 300
MULTI_DAY_TICK_MIN = 60
MULTI_DAY_TARGET_MAX_BINS = 360
MULTI_DAY_TIME_FORMAT = "%m-%d %H:%M"


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s)


def _overlay_warning_text(ax, *, t0: datetime, bin_seconds: int, num_bins: int,
                          warns: 'Sequence[Any]', group_labels: bool, label_groups: Dict[str, List[str]] | Dict[str, set]) -> None:
    """Overlay compact warning text near vertical lines to indicate cause/group."""
    if not warns or num_bins <= 0:
        return
    color_map = {
        WarningType.ALERT: "#ff2d2d",
        WarningType.CLUSTER: "#ffaa00",
        WarningType.LOUDNESS: "#3399ff",
        WarningType.INFORMATIONAL: "#aaaaaa",
    }
    prev_b: Optional[int] = None
    marker_y = 0.0
    for w in warns:
        b = int((w.timestamp - t0).total_seconds() // max(1, bin_seconds))
        if not (0 <= b < num_bins):
            continue
        warning_groups = None
        warning_cause = getattr(w, "label", None)
        if group_labels and warning_cause:
            warning_groups = [g for g, members in label_groups.items() if warning_cause in members]

        # Compose text
        if w.warning_type == WarningType.ALERT:
            marker_text = "ALERT!\n"
        elif w.warning_type == WarningType.CLUSTER:
            marker_text = "warning:\n" + w.warning_type.value
            if warning_groups:
                marker_text += ": " + ",".join(warning_groups)
            elif warning_cause:
                marker_text += ": " + str(warning_cause)
        else:
            marker_text = "warning:\n" + w.warning_type.value

        # Adjust vertical stacking if too close to previous
        if prev_b is not None and (b - prev_b) < 10:
            marker_y += 0.5
        else:
            marker_y = 0.0
        prev_b = b

        ax.text(
            b + 0.5,
            marker_y,
            marker_text,
            rotation=0,
            ha='center',
            va='top',
            color='white',
            fontsize=7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='none', alpha=0.6),
        )


def plot_multi_day_overview(ds: DataStore,
                            viz: Visualizer,
                            label_groups: Dict[str, List[str]] | Dict[str, set],
                            *,
                            days: int = MULTI_DAY_DAYS,
                            end_at: Optional[datetime] = None,
                            bin_seconds: int = MULTI_DAY_BIN_SECONDS,
                            tick_minutes: int = MULTI_DAY_TICK_MIN,
                            group_labels: bool = True,
                            target_max_bins: int = MULTI_DAY_TARGET_MAX_BINS,
                            time_format: str = MULTI_DAY_TIME_FORMAT) -> None:
    # Determine default window from DB without loading rows
    full_start, full_end = ds.time_bounds()
    if not full_start or not full_end:
        print("No events; nothing to plot.")
        return
    if end_at is None:
        end_at = full_end
    start_at = max(full_start, end_at - timedelta(days=days))
    if start_at >= end_at:
        print("Invalid period window for multi‑day overview.")
        return

    # Build matrix via Visualizer builder
    mat, labels, meta = viz.build_heatmap_matrix(
        start=start_at,
        end=end_at,
        bin_seconds=bin_seconds,
        group_labels=group_labels,
        label_groups=label_groups if group_labels else None,
        min_total_seconds=MIN_TOTAL_SECONDS,
        top_n=TOP_N,
        max_bins=target_max_bins,
    )
    if not labels or meta.get("num_bins", 0) == 0:
        print("No data in requested multi‑day period.")
        return

    # Plot using Visualizer's renderer
    fig = viz.plot_heatmap(
        mat,
        labels,
        meta,
        tick_minutes=tick_minutes,
        time_format=time_format,
        show_warnings=True,
        show_loudness=True,
        max_cell_seconds=MAX_CELL_SECONDS,
    )

    # Overlay warning text markers on the main axis
    try:
        warns = viz.warnings_for_window(start=meta["t0"], end=meta["t_end"], types=[WarningType.ALERT, WarningType.CLUSTER, WarningType.LOUDNESS])
    except Exception:
        warns = []
    if fig.axes:
        ax_main = fig.axes[0]
        _overlay_warning_text(
            ax_main,
            t0=meta["t0"],
            bin_seconds=int(meta["bin_seconds"]),
            num_bins=int(meta["num_bins"]),
            warns=warns,
            group_labels=group_labels,
            label_groups=label_groups,
        )

    plt.tight_layout()
    plt.show()


def main() -> int:
    cfg = load_metamodel_config()
    label_groups = cfg.label_groups

    ds = DataStore(DB_PATH)
    viz = Visualizer(ds, default_snippet_seconds=float(cfg.model_parameters.get("default_snippet_duration_seconds", 1.0)),
                     merge_gap_seconds=float(cfg.model_parameters.get("merge_gap_seconds", 3.0)))

    # Determine default bounds without loading all events
    full_start, full_end = ds.time_bounds()
    if not full_start or not full_end:
        print("No events in DB; nothing to visualize.")
        return 0

    # Window selection
    win_end = _parse_iso(WINDOW_END_ISO) or full_end
    if WINDOW_LAST_SECONDS and WINDOW_LAST_SECONDS > 0:
        win_start = win_end - timedelta(seconds=WINDOW_LAST_SECONDS)
    else:
        win_start = _parse_iso(WINDOW_START_ISO) or full_start
    if win_start >= win_end:
        print("Invalid window (start >= end)")
        return 0

    if MULTI_DAY_ENABLED:
        plot_multi_day_overview(ds, viz, label_groups,
                                days=MULTI_DAY_DAYS,
                                end_at=full_end,
                                bin_seconds=MULTI_DAY_BIN_SECONDS,
                                tick_minutes=MULTI_DAY_TICK_MIN,
                                group_labels=GROUP_LABELS,
                                target_max_bins=MULTI_DAY_TARGET_MAX_BINS,
                                time_format=MULTI_DAY_TIME_FORMAT)
        return 0

    # Fast existence check without loading rows
    if ds.count_in_range(start=win_start, end=win_end) == 0:
        print("No events inside requested window.")
        return 0

    # Build matrix using Visualizer builder
    mat, labels, meta = viz.build_heatmap_matrix(
        start=win_start,
        end=win_end,
        bin_seconds=BIN_SECONDS,
        group_labels=GROUP_LABELS,
        label_groups=label_groups if GROUP_LABELS else None,
        min_total_seconds=MIN_TOTAL_SECONDS,
        top_n=TOP_N,
        max_bins=MAX_BINS,
    )
    if not labels or meta.get("num_bins", 0) == 0:
        print("No data in requested window.")
        return 0

    # Render using Visualizer's plotter
    fig = viz.plot_heatmap(
        mat,
        labels,
        meta,
        tick_minutes=TICK_MINUTES,
        time_format=TIME_FORMAT,
        show_warnings=True,
        show_loudness=True,
        max_cell_seconds=MAX_CELL_SECONDS,
    )

    # Alternative: Plot with Plotly renderer
    # fig = viz.plot_heatmap_plotly(
    #     mat,
    #     labels,
    #     meta,
    #     show_warnings=True,
    #     annotate_warning_labels=True,
    #     show_loudness=True,
    #     max_cell_seconds=MAX_CELL_SECONDS,
    #     group_labels=GROUP_LABELS,
    #     label_groups=label_groups,
    #     loud_min_std=float(getattr(cfg, "loudness_min_std", 0.05)),
    #     loud_sigma_mult=float(getattr(cfg, "loudness_sigma_multiplier", 3.0)),
    # )

    # fig.show()

    # Overlay warning text markers on the main axis
    try:
        warns = viz.warnings_for_window(start=meta["t0"], end=meta["t_end"], types=[WarningType.ALERT, WarningType.CLUSTER, WarningType.LOUDNESS])
    except Exception:
        warns = []
    if fig.axes:
        ax_main = fig.axes[0]
        _overlay_warning_text(
            ax_main,
            t0=meta["t0"],
            bin_seconds=int(meta["bin_seconds"]),
            num_bins=int(meta["num_bins"]),
            warns=warns,
            group_labels=GROUP_LABELS,
            label_groups=label_groups,
        )

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
