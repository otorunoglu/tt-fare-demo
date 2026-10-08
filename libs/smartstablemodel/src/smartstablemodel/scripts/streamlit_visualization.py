import os
import streamlit as st
from datetime import datetime, timedelta, date, time
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from smartstablemodel.services.data_store import DataStore
from smartstablemodel.services.visualizer import Visualizer
from smartstablemodel.config import load_metamodel_config

DB_PATH = os.getenv("SMARTSTABLE_DB_PATH") or ""
if not DB_PATH:
    raise RuntimeError("SMARTSTABLE_DB_PATH must be set")
DEFAULT_BIN_MINUTES = 2
DEFAULT_WINDOW_HOURS = 12
DEFAULT_START_TIME = time(19, 0)  # 19:00 evening
DEFAULT_END_TIME = time(9, 0)  # 09:00 morning

group_labels: bool = False
cfg = load_metamodel_config()
label_groups = cfg.label_groups


def _coerce_dt(d: Any, default_time: datetime) -> datetime:
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        return datetime.combine(d, default_time.time())
    return default_time


def _resolve_window(
    start_in: date,
    end_in: date,
    start_time: time,
    end_time: time,
) -> tuple[datetime, datetime]:
    start = datetime.combine(start_in, start_time)
    end = datetime.combine(end_in, end_time)

    if end <= start and end_in == start_in:
        end += timedelta(days=1)

    return start, end


def _daily_time_rangebreaks(
    *, start_time: time, end_time: time, spans_multiple_days: bool
) -> list[dict[str, Any]]:
    start_dec = start_time.hour + start_time.minute / 60.0
    end_dec = end_time.hour + end_time.minute / 60.0

    if start_dec == end_dec:
        return []
    if end_dec < start_dec:
        return [dict(bounds=[end_dec, start_dec], pattern="hour")]
    if spans_multiple_days:
        return [
            dict(bounds=[0, start_dec], pattern="hour"),
            dict(bounds=[end_dec, 24], pattern="hour"),
        ]
    return []


def _daily_block_separators(
    *, start: datetime, end: datetime, end_time: time
) -> list[datetime]:
    separators: list[datetime] = []
    cur_day = start.date()
    last_day = end.date()

    while cur_day <= last_day:
        boundary = datetime.combine(cur_day, end_time)
        if start < boundary < end:
            separators.append(boundary)
        cur_day += timedelta(days=1)

    return separators


def build_warnings_df(
    viz: Visualizer, *, start: datetime, end: datetime
) -> pd.DataFrame:
    warns = viz.warnings_for_window(start=start, end=end, types=None)  # all types
    if not warns:
        return pd.DataFrame(columns=["time", "type", "label", "group"])
    rows = []
    for w in warns:
        ts = (
            w.timestamp
            if isinstance(w.timestamp, datetime)
            else datetime.fromisoformat(str(w.timestamp))
        )
        rows.append(
            {
                "time": ts,
                "type": getattr(w.warning_type, "value", str(w.warning_type)),
                "label": getattr(w, "label", None),
                "group": getattr(w, "group", None) or getattr(w, "label", None),
            }
        )
    return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def main():
    global group_labels
    st.set_page_config(page_title="Soundscape Visualization", layout="wide")

    ds = DataStore(DB_PATH)
    viz = Visualizer(ds)

    full_start, full_end = ds.time_bounds()
    if not full_start:
        st.warning("No events in database.")
        return
    if not full_end:
        full_end = full_start + timedelta(hours=DEFAULT_WINDOW_HOURS)

    # For overnight monitoring (18:00-09:00), default start date is yesterday, end date is today
    default_start_date = (full_end - timedelta(days=1)).date()
    default_end_date = full_end.date()

    # Get dates with recordings to help user selection
    available_dates = ds.dates_with_recordings()
    available_dates_set = set(available_dates)

    col1, col2, col3 = st.columns(3)
    with col1:
        start_in = st.date_input("Start date", value=default_start_date)
        # Indicate if selected date has recordings
        if start_in in available_dates_set:
            st.caption("✅ Has recordings")
        else:
            st.caption("⚠️ No recordings on this date")
    with col2:
        end_in = st.date_input("End date", value=default_end_date)
        if end_in in available_dates_set:
            st.caption("✅ Has recordings")
        else:
            st.caption("⚠️ No recordings on this date")
    with col3:
        bin_size_minutes = st.select_slider(
            "Bin size (minutes)",
            options=[1, 2, 5, 10, 15, 30, 60],
            value=DEFAULT_BIN_MINUTES,
        )

    # Show available date range with recordings
    if available_dates:
        st.caption(
            f"📅 Recordings available: {available_dates[0].strftime('%Y-%m-%d')} to {available_dates[-1].strftime('%Y-%m-%d')} ({len(available_dates)} days)"
        )

    # Time range inputs for nighttime filtering
    time_col1, time_col2, time_col3 = st.columns(3)
    with time_col1:
        start_time = st.time_input("Start time", value=DEFAULT_START_TIME)
    with time_col2:
        end_time = st.time_input("End time", value=DEFAULT_END_TIME)
    with time_col3:
        hide_empty_bins = st.checkbox(
            "Hide empty time bins",
            value=True,
            help="Remove time bins where all labels have zero events",
        )

    group_labels = st.checkbox("Group labels", value=False)
    cap = st.select_slider(
        "Cap minutes per bin",
        options=[0.15, 1, 2, 5, 10, 15, 30, 60],
        value=1,
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Events below this confidence are reassigned to 'background' for visualization.",
    )

    # Combine date and time inputs.
    start, end = _resolve_window(start_in, end_in, start_time, end_time)

    # Reject backwards ranges that span multiple explicit dates.
    if end <= start:
        st.error(
            "End must be after start. For overnight windows on the same date, choose an end time after midnight."
        )
        return

    # Build heatmap matrix directly via Visualizer (full set first for label discovery)
    mat_full, labels_full, meta = viz.build_heatmap_matrix(
        start=start,
        end=end,
        bin_seconds=bin_size_minutes * 60,
        confidence_threshold=float(confidence_threshold),
        group_labels=group_labels,
        label_groups=getattr(cfg, "label_groups", None),
        min_total_seconds=0.0,
        top_n=None,
        max_bins=2000,
    )

    if (
        int(meta.get("num_bins", 0)) == 0
        or not labels_full
        or getattr(mat_full, "size", 0) == 0
    ):
        st.info("No bins available for the current selection.")
        return

    # Interactive label filter: multiselect with all labels selected by default
    selected_labels = st.multiselect(
        "Filter labels/groups",
        options=labels_full,
        default=labels_full,
        help="Deselect labels to hide them from the heatmap.",
    )
    if not selected_labels:
        st.warning("Select at least one label to display.")
        return

    # Filter matrix and labels to selected subset (preserve order from labels_full)
    keep_idx = [i for i, lab in enumerate(labels_full) if lab in selected_labels]
    mat = mat_full[keep_idx, :] if len(keep_idx) < len(labels_full) else mat_full
    labels = [labels_full[i] for i in keep_idx]

    # Calculate rangebreaks to hide hours outside the selected daily window.
    rbreaks = []
    if hide_empty_bins:
        rbreaks = _daily_time_rangebreaks(
            start_time=start_time,
            end_time=end_time,
            spans_multiple_days=end_in > start_in,
        )

    if start != end:
        st.caption(f"Showing data from **{start}** to **{end}**")

    if hide_empty_bins and not rbreaks:
        non_empty_cols = np.where(mat.sum(axis=0) > 0)[0]
        if len(non_empty_cols) == 0:
            st.info("All time bins are empty for the selected labels.")
            return
        mat = mat[:, non_empty_cols]
        # Update meta to match filtered bins - create a copy to avoid mutating original
        meta = dict(meta)
        meta["num_bins"] = len(non_empty_cols)
        # Update x_labels if present
        if "x_labels" in meta and meta["x_labels"] is not None:
            meta["x_labels"] = [meta["x_labels"][i] for i in non_empty_cols]
        # Recompute t0 and t_end based on actual remaining bins for proper x-axis range
        bin_seconds = int(meta["bin_seconds"])
        original_t0 = meta["t0"]
        # Calculate actual time positions for remaining bins
        kept_times = [
            original_t0 + timedelta(seconds=int(i) * bin_seconds)
            for i in non_empty_cols
        ]
        meta["t0"] = kept_times[0]
        meta["t_end"] = kept_times[-1] + timedelta(seconds=bin_seconds)
        # Store the actual kept bin indices so visualizer can use correct times
        meta["_kept_bin_times"] = kept_times

    # Render with centralized Plotly renderer in Visualizer
    fig = viz.plot_heatmap_plotly(
        mat,
        labels,
        meta,
        show_warnings=True,
        annotate_warning_labels=True,
        show_loudness=True,
        max_cell_seconds=float(cap * 60),
        group_labels=group_labels,
        label_groups=label_groups,
        loud_min_std=float(getattr(cfg, "loudness_min_std", 0.05)),
        loud_sigma_mult=float(getattr(cfg, "loudness_sigma_multiplier", 3.0)),
        rangebreaks=rbreaks,
        show_midnight_separators=end_time < start_time,
    )

    for separator in _daily_block_separators(start=start, end=end, end_time=end_time):
        fig.add_vline(
            x=separator,
            line_dash="dot",
            line_color="rgba(0, 0, 0, 0.45)",
            line_width=1.5,
        )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
