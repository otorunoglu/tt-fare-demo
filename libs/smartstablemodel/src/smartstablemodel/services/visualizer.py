"""
DB-based visualization utilities decoupled from the MetaModelDecider.

Provides grouped span computation and convenience accessors reading directly from DataStore.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Sequence, Mapping

import numpy as np
from smartstablemodel.domain.warning import WarningType

from .data_store import DataStore, DomainWarning


class Visualizer:
    def __init__(
        self,
        datastore: DataStore,
        *,
        default_snippet_seconds: float = 1.0,
        merge_gap_seconds: float = 3.0,
    ) -> None:
        self.datastore = datastore
        self.default_snippet_seconds = float(default_snippet_seconds)
        self.merge_gap_seconds = float(merge_gap_seconds)

    def events(
        self, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in self.datastore.query(start=start, end=end)]

    def grouped_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        *,
        merge_gap_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        merge_gap = (
            float(merge_gap_seconds)
            if merge_gap_seconds is not None
            else self.merge_gap_seconds
        )
        events = self.datastore.query(start=start, end=end)
        if not events:
            return []

        open_spans: Dict[str, Tuple[datetime, datetime]] = {}
        closed: List[Tuple[str, datetime, datetime]] = []
        seg_dur = self.default_snippet_seconds
        for ev in events:
            lab = getattr(ev, "label", None)
            if not lab:
                continue
            ts = ev.timestamp
            ev_start = ts
            ev_end = ts + timedelta(seconds=seg_dur)
            if lab not in open_spans:
                open_spans[lab] = (ev_start, ev_end)
                continue
            cur_start, cur_end = open_spans[lab]
            gap = (ev_start - cur_end).total_seconds()
            if gap <= merge_gap:
                new_end = ev_end if ev_end > cur_end else cur_end
                open_spans[lab] = (cur_start, new_end)
            else:
                if cur_end > cur_start:
                    closed.append((lab, cur_start, cur_end))
                open_spans[lab] = (ev_start, ev_end)
        for lab, (s, e) in open_spans.items():
            if e > s:
                closed.append((lab, s, e))

        grouped: List[Dict[str, Any]] = []
        for lab, s, e in closed:
            seconds = (e - s).total_seconds()
            if seconds <= 0:
                continue
            grouped.append({"label": lab, "start": s, "end": e, "seconds": seconds})
        grouped.sort(key=lambda d: d["start"])  # type: ignore[index]
        return grouped

    # --------------------------- DB-backed helpers ---------------------------
    def label_bin_counts(
        self,
        *,
        start: datetime,
        end: datetime,
        bin_seconds: int,
        confidence_threshold: float = 0.0,
        reassign_low_confidence_to_background: bool = True,
    ) -> Dict[str, Dict[int, int]]:
        """Return per-label bin counts using DataStore SQL aggregation."""
        return self.datastore.bin_counts_by_label(
            start=start,
            end=end,
            bin_seconds=bin_seconds,
            confidence_threshold=confidence_threshold,
            reassign_low_confidence_to_background=reassign_low_confidence_to_background,
        )

    def group_bin_counts(
        self,
        label_groups: "Mapping[str, Sequence[str]] | Mapping[str, set] | Dict[str, Sequence[str]] | Dict[str, set]",
        label_bin_counts: Dict[str, Dict[int, int]],
    ) -> Dict[str, Dict[int, int]]:
        """Aggregate label bin counts into group bin counts based on label_groups mapping."""
        label_to_groups: Dict[str, set[str]] = {}
        for gname, members in label_groups.items():
            for m in members:
                label_to_groups.setdefault(m, set()).add(str(gname))
        group_bins: Dict[str, Dict[int, int]] = {}
        for lab, bins in label_bin_counts.items():
            groups = label_to_groups.get(lab)
            if not groups:
                continue
            for b, c in bins.items():
                for g in groups:
                    group_bins.setdefault(g, {})[b] = group_bins.get(g, {}).get(
                        b, 0
                    ) + int(c)
        return group_bins

    def to_matrix(
        self,
        bins_by_row: Dict[str, Dict[int, int]],
        *,
        num_bins: int,
        min_total_seconds: float = 0.0,
        top_n: Optional[int] = None,
        text_by_row: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
        """
        Convert a mapping row->bin->seconds into (ordered labels, matrix[rows,num_bins], text_matrix).
        text_by_row: {label: {bin_idx: "text"}}
        """
        totals: Dict[str, float] = {
            k: float(sum(v.values())) for k, v in bins_by_row.items()
        }
        labels = [k for k, tot in totals.items() if tot >= float(min_total_seconds)]
        labels.sort(key=lambda k: -totals[k])
        if top_n and len(labels) > int(top_n):
            labels = labels[: int(top_n)]

        mat = np.zeros((len(labels), int(num_bins)), dtype=float)
        # Use object array for text to support variable length strings/None
        text_mat = np.full((len(labels), int(num_bins)), None, dtype=object)

        idx = {lab: i for i, lab in enumerate(labels)}

        for lab, bins in bins_by_row.items():
            if lab not in idx:
                continue
            row = idx[lab]
            for b, c in bins.items():
                if 0 <= int(b) < int(num_bins):
                    mat[row, int(b)] += float(c)

        if text_by_row:
            for lab, texts in text_by_row.items():
                if lab not in idx:
                    continue
                row = idx[lab]
                for b, txt in texts.items():
                    if 0 <= int(b) < int(num_bins):
                        text_mat[row, int(b)] = txt

        return labels, mat, text_mat

    def loudness_avg_bins(
        self, *, start: datetime, end: datetime, bin_seconds: int
    ) -> Dict[int, float]:
        """
        Return average loudness per bin index in [start,end) using DB aggregation if available.
        """
        # Prefer DataStore fast SQL path
        if hasattr(self.datastore, "bin_avg_loudness"):
            return self.datastore.bin_avg_loudness(
                start=start, end=end, bin_seconds=bin_seconds
            )
        # Fallback: return empty (prevents crashes; plot will just skip loudness)
        return {}

    def loudness_stats(
        self, *, start: datetime, end: datetime
    ) -> Tuple[int, float, float]:
        """
        Return (n, mean, std) loudness across [start,end) using DB if available.
        """
        if hasattr(self.datastore, "loudness_stats"):
            return self.datastore.loudness_stats(start=start, end=end)
        # Fallback: no data
        return (0, 0.0, 0.0)

    def warnings_for_window(
        self,
        *,
        start: datetime,
        end: datetime,
        types: Optional[Sequence[WarningType]] = None,
    ) -> Sequence[DomainWarning]:
        """
        Return persisted warnings in [start,end). Delegates to DataStore if available.
        """
        if hasattr(self.datastore, "warnings"):
            return self.datastore.warnings(start=start, end=end, types=types)
        return []

    def reason_roll(
        self, mat: np.ndarray, *, bin_seconds: int, burst_window_seconds: int
    ) -> np.ndarray:
        """
        Compute rolling coverage over `burst_window_seconds` along the time axis for each row.
        Returns a matrix of same shape as mat.
        """
        if mat.size == 0:
            return mat
        window_bins = max(1, int(round(burst_window_seconds / float(bin_seconds or 1))))
        kernel = np.ones(window_bins, dtype=float)
        roll = np.zeros_like(mat, dtype=float)
        for i in range(mat.shape[0]):
            conv = np.convolve(mat[i], kernel, mode="full")
            roll[i] = conv[window_bins - 1 : window_bins - 1 + mat.shape[1]]
        return roll

    # --------------------------- High-level builders ---------------------------
    def _ensure_bins(
        self,
        t0: datetime,
        t_end: datetime,
        bin_seconds: int,
        limit: Optional[int] = None,
    ) -> Tuple[int, int]:
        span = max(0.0, (t_end - t0).total_seconds())
        est = int(np.ceil(span / bin_seconds)) if bin_seconds > 0 else 1
        if limit and est > limit:
            factor = int(np.ceil(est / limit))
            bin_seconds *= max(1, factor)
            est = int(np.ceil(span / bin_seconds))
        return est, bin_seconds

    def build_heatmap_matrix(
        self,
        start: datetime,
        end: datetime,
        *,
        bin_seconds: int = 120,
        confidence_threshold: float = 0.0,
        reassign_low_confidence_to_background: bool = True,
        group_labels: bool = False,
        label_groups: Optional[Mapping[str, Sequence[str]] | Mapping[str, set]] = None,
        min_total_seconds: float = 0.0,
        top_n: Optional[int] = None,
        max_bins: Optional[int] = 1000,
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Build a heatmap matrix (rows x bins) and metadata for plotting.
        Returns (matrix, labels, meta) where meta includes t0, t_end, num_bins, bin_seconds, y_label.
        """
        # Coerce dates to datetimes at midnight if needed
        if not isinstance(start, datetime):
            start = datetime.combine(start, datetime.min.time())  # type: ignore[arg-type]
        if not isinstance(end, datetime):
            end = datetime.combine(end, datetime.min.time())  # type: ignore[arg-type]

        t0, t_end = start, end
        num_bins, bin_seconds = self._ensure_bins(t0, t_end, bin_seconds, max_bins)
        if self.datastore.count_in_range(start=t0, end=t_end) == 0:
            return (
                np.zeros((0, 0), dtype=float),
                [],
                {
                    "t0": t0,
                    "t_end": t_end,
                    "num_bins": 0,
                    "bin_seconds": bin_seconds,
                    "y_label": "",
                },
            )

        label_bins = self.label_bin_counts(
            start=t0,
            end=t_end,
            bin_seconds=bin_seconds,
            confidence_threshold=confidence_threshold,
            reassign_low_confidence_to_background=reassign_low_confidence_to_background,
        )

        # Fetch extra text details (e.g. original labels for anomalies)
        # Assuming DataStore has this method now (we added it)
        text_by_row = {}
        if hasattr(self.datastore, "anomaly_details_by_bin"):
            text_by_row = self.datastore.anomaly_details_by_bin(
                start=t0, end=t_end, bin_seconds=bin_seconds
            )

        if group_labels:
            if not label_groups:
                raise ValueError("label_groups must be provided when group_labels=True")
            effective_groups = dict(label_groups)
            if "background" in label_bins and not any(
                "background" in members for members in effective_groups.values()
            ):
                effective_groups["background"] = ["background"]

            group_bins = self.group_bin_counts(effective_groups, label_bins)
            # We don't easily support text grouping yet, so pass None or try to map?
            # For now, if grouping is on, we might lose the granular anomaly text unless we map it to the group 'anomaly_labels'
            # Let's try to map it if the anomaly label is in a group
            grouped_text = {}
            if text_by_row:
                # Invert group map
                lab_to_group = {}
                for g, members in label_groups.items():
                    for m in members:
                        lab_to_group[m] = g

                for lab, txts in text_by_row.items():
                    g = lab_to_group.get(lab)
                    if g:
                        # Append text? For now just overwrite or use first
                        grouped_text[g] = txts

            labels, mat, text_mat = self.to_matrix(
                group_bins,
                num_bins=num_bins,
                min_total_seconds=min_total_seconds,
                top_n=top_n,
                text_by_row=grouped_text,
            )
            y_label = "Group"
        else:
            labels, mat, text_mat = self.to_matrix(
                label_bins,
                num_bins=num_bins,
                min_total_seconds=min_total_seconds,
                top_n=top_n,
                text_by_row=text_by_row,
            )
            y_label = "Label"

        meta: Dict[str, Any] = {
            "t0": t0,
            "t_end": t_end,
            "num_bins": num_bins,
            "bin_seconds": bin_seconds,
            "y_label": y_label,
            "text_matrix": text_mat,  # Pack it for the plotter
        }
        return mat, labels, meta

    def plot_heatmap(
        self,
        mat: np.ndarray,
        labels: List[str],
        meta: Dict[str, Any],
        *,
        tick_minutes: int = 15,
        time_format: str = "%H:%M",
        show_warnings: bool = True,
        show_loudness: bool = True,
        max_cell_seconds: Optional[float] = None,
        use_seaborn: bool = False,
    ):
        """
        Render a Matplotlib heatmap (with optional loudness subplot) using the matrix and meta from build_heatmap_matrix.
        Returns a Matplotlib figure. Import is local to avoid a hard dependency for non-plot callers.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib import cm
        from smartstablemodel.domain.warning import WarningType

        t0 = meta["t0"]
        t_end = meta["t_end"]
        num_bins = int(meta["num_bins"])
        bin_seconds = int(meta["bin_seconds"])
        y_label = meta.get("y_label", "")
        yticklabels = labels

        # Figure and axis setup
        fig_h = max(4, 0.45 * (len(yticklabels)) + 1.2)
        fig_w = min(22, 1.0 + 0.14 * max(1, num_bins))
        gs = None
        if show_loudness:
            fig = plt.figure(figsize=(fig_w, fig_h + 2.5))
            gs = fig.add_gridspec(
                2, 1, height_ratios=[max(1, fig_h - 1.5), 2.2], hspace=0.25
            )
            ax = fig.add_subplot(gs[0])
        else:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Color normalization (support optional capping like the CLI script)
        if max_cell_seconds is not None and float(max_cell_seconds) > 0:
            norm = mcolors.Normalize(vmin=0, vmax=float(max_cell_seconds), clip=False)
            extend_flag = "max"
        else:
            vmax = mat.max() if mat.size else 1.0
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            extend_flag = "neither"

        # Render heatmap: either Seaborn (with cell gridlines) or plain Matplotlib
        if use_seaborn:
            try:
                import seaborn as sns  # local import; optional dependency

                sns.heatmap(
                    mat,
                    ax=ax,
                    cmap="magma",
                    norm=norm,
                    cbar=False,  # create a consistent colorbar below
                    linewidths=0.25,
                    linecolor="white",
                    square=False,
                )
            except ImportError:
                ax.imshow(
                    mat, aspect="auto", interpolation="nearest", cmap="magma", norm=norm
                )
        else:
            ax.imshow(
                mat, aspect="auto", interpolation="nearest", cmap="magma", norm=norm
            )

        # Colorbar (consistent for both paths)
        sm = cm.ScalarMappable(norm=norm, cmap="magma")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, extend=extend_flag)
        cbar.set_label("Seconds")

        # X ticks
        tick_seconds = tick_minutes * 60
        tick_positions: List[int] = []
        tick_labels: List[str] = []
        cur = t0
        while cur <= t_end and num_bins > 0:
            b = int((cur - t0).total_seconds() // bin_seconds)
            if 0 <= b < num_bins:
                tick_positions.append(b)
                tick_labels.append(cur.strftime(time_format))
            cur += timedelta(seconds=tick_seconds)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel(y_label)
        ax.set_title("Duration heatmap")

        # Warnings overlay
        if show_warnings and num_bins > 0:
            try:
                warns = self.warnings_for_window(
                    start=t0,
                    end=t_end,
                    types=[
                        WarningType.ALERT,
                        WarningType.CLUSTER,
                        WarningType.LOUDNESS,
                    ],
                )
            except Exception:
                warns = []
            color_map = {
                WarningType.ALERT: "#ff2d2d",
                WarningType.CLUSTER: "#ffaa00",
                WarningType.LOUDNESS: "#3399ff",
                WarningType.INFORMATIONAL: "#aaaaaa",
            }
            for w in warns:
                b = int((w.timestamp - t0).total_seconds() // bin_seconds)
                if 0 <= b < num_bins:
                    ax.axvline(
                        b + 0.5,
                        color=color_map.get(w.warning_type, "#ffffff"),
                        linewidth=1.1,
                        alpha=0.85,
                    )

        # Loudness subplot
        if show_loudness and gs is not None:
            ax2 = fig.add_subplot(gs[1], sharex=ax)
            loud_bins = self.loudness_avg_bins(
                start=t0, end=t_end, bin_seconds=bin_seconds
            )
            xs = list(range(num_bins))
            ys = [loud_bins.get(b, np.nan) for b in xs]
            ax2.plot(xs, ys, color="#3fa7ff", linewidth=1.2, label="avg loudness")
            n, mu, sigma = self.loudness_stats(start=t0, end=t_end)
            # Choose defaults similar to CLI
            sigma_mult = 3.0
            min_std = 0.05
            eff_sigma = max(sigma, min_std)
            thr = mu + sigma_mult * eff_sigma
            ax2.axhline(
                mu,
                color="#77cc77",
                linestyle="--",
                linewidth=1.0,
                label=f"baseline μ={mu:.2f}",
            )
            ax2.axhline(
                thr,
                color="#ff7733",
                linestyle=":",
                linewidth=1.0,
                label=f"threshold μ+{sigma_mult}σ",
            )
            ax2.set_ylabel("Loudness")
            ax2.set_xlabel(f"Time (bin={bin_seconds}s)")
            ax2.grid(True, axis="y", alpha=0.2)
            ax2.legend(loc="upper left", fontsize=8, frameon=True)

        plt.tight_layout()
        return fig

    def plot_heatmap_plotly(
        self,
        mat: np.ndarray,
        labels: List[str],
        meta: Dict[str, Any],
        *,
        show_warnings: bool = True,
        annotate_warning_labels: bool = True,
        show_loudness: bool = True,
        max_cell_seconds: Optional[float] = None,
        group_labels: bool = False,
        label_groups: Optional[Mapping[str, Sequence[str]] | Mapping[str, set]] = None,
        loud_min_std: float = 0.05,
        loud_sigma_mult: float = 3.0,
        rangebreaks: Optional[List[Dict[str, Any]]] = None,
        show_midnight_separators: bool = True,
    ):
        """
        Render a Plotly heatmap (with optional loudness subplot) using the matrix and meta from build_heatmap_matrix.
        Returns a Plotly Figure. Imports are local to keep optional dependency.
        """
        from plotly.subplots import make_subplots  # type: ignore
        import plotly.graph_objects as go  # type: ignore

        t0 = meta["t0"]
        t_end = meta["t_end"]
        num_bins = int(meta["num_bins"])
        bin_seconds = int(meta["bin_seconds"])
        y_label = meta.get("y_label", "")

        # Use explicit bin times if provided (for filtered/non-contiguous bins)
        if "_kept_bin_times" in meta and meta["_kept_bin_times"] is not None:
            times = meta["_kept_bin_times"]
        else:
            times = [
                t0 + timedelta(seconds=i * bin_seconds) for i in range(max(0, num_bins))
            ]
        z = mat.copy() if isinstance(mat, np.ndarray) else np.array(mat)
        if max_cell_seconds is not None:
            z = np.minimum(z, float(max_cell_seconds))

        # Extract text matrix if present
        text_mat = meta.get("text_matrix")
        if text_mat is None:
            # Create empty for Hovertemplate to be safe
            text_mat = np.full_like(z, None, dtype=object)

        heat_h = max(200, 28 * max(1, len(labels)))
        loud_h = 220 if show_loudness else 0
        rows = 2 if show_loudness else 1
        row_heights = [heat_h, loud_h] if show_loudness else [heat_h]
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07 if show_loudness else 0.02,
            row_heights=row_heights,
        )

        heat = go.Heatmap(
            z=z,
            x=times,
            y=labels,
            text=text_mat,
            colorscale="YlGnBu",
            zmin=0,
            zmax=float(max_cell_seconds) if max_cell_seconds is not None else None,
            colorbar=dict(title="sec/bin"),
            hovertemplate="%{y}<br>%{x|%H:%M:%S}<br>sec=%{z}<extra></extra>",
        )
        fig.add_trace(heat, row=1, col=1)
        fig.update_yaxes(
            title_text=y_label or "Label", row=1, col=1, autorange="reversed"
        )
        fig.update_xaxes(title_text="Time", row=1, col=1, rangebreaks=rangebreaks)

        if show_midnight_separators:
            # Add vertical line at midnight for each day in range.
            start_midnight = datetime.combine(t0.date(), datetime.min.time())
            if start_midnight < t0:
                start_midnight += timedelta(days=1)

            curr_midnight = start_midnight
            while curr_midnight <= t_end:
                fig.add_vline(
                    x=curr_midnight,
                    line_dash="dash",
                    line_color="black",
                    opacity=0.8,
                    line_width=2,
                )
                curr_midnight += timedelta(days=1)

        if show_warnings and num_bins > 0:
            # --- SCATTER OVERLAY FOR ANOMALIES ---
            # Find row index for 'anomaly_abnormal'
            anomaly_label = "anomaly_abnormal"
            if anomaly_label in labels:
                try:
                    row_idx = labels.index(anomaly_label)
                    # Find all bins where value > 0
                    # z is (rows, cols)
                    row_data = z[row_idx, :]
                    nonzero_cols = np.where(row_data > 0)[0]

                    if len(nonzero_cols) > 0:
                        # Map cols to times
                        # times is list of datetimes corresponding to cols
                        scatter_x = [times[c] for c in nonzero_cols]
                        scatter_y = [anomaly_label] * len(nonzero_cols)
                        # Optional: pull custom text if available
                        scatter_text = None
                        point_colors = []
                        if text_mat is not None:
                            scatter_text = [text_mat[row_idx, c] for c in nonzero_cols]
                            # Determine color: Red for true, Grey for false alarm
                            for txt in scatter_text:
                                t_lower = str(txt).lower() if txt else ""
                                if (
                                    "false alarm" in t_lower
                                    or "confirmed:false" in t_lower
                                ):
                                    point_colors.append("#888888")  # Grey
                                else:
                                    point_colors.append("#ff0000")  # Red
                        else:
                            # Default to red if no text details
                            point_colors = ["#ff0000"] * len(scatter_x)

                        fig.add_trace(
                            go.Scatter(
                                x=scatter_x,
                                y=scatter_y,
                                mode="markers",
                                marker=dict(
                                    symbol="diamond",
                                    color=point_colors,
                                    size=8,
                                    line=dict(width=1, color="white"),
                                ),
                                name="Anomaly Marker",
                                hovertemplate="%{y}<br>%{x|%H:%M:%S}<br>%{text}<extra></extra>",
                                text=scatter_text if scatter_text else None,
                                showlegend=False,  # Cleaner
                            ),
                            row=1,
                            col=1,
                        )
                except Exception:
                    pass

            color_map = {
                WarningType.ALERT: "#ff2d2d",
                WarningType.CLUSTER: "#ffaa00",
                WarningType.LOUDNESS: "#3399ff",
                WarningType.INFORMATIONAL: "#aaaaaa",
            }

            try:
                warns = self.warnings_for_window(
                    start=t0,
                    end=t_end,
                    types=[
                        WarningType.ALERT,
                        WarningType.CLUSTER,
                        WarningType.LOUDNESS,
                    ],
                )
            except Exception:
                warns = []
            prev_time: Optional[datetime] = None
            prev_warning: Optional[WarningType] = None
            group_with_prev: bool = False
            warning_counter = 1
            severity_sum = 0.0
            marker_y = 0.00

            grouped_warnings: List[Any] = []
            if annotate_warning_labels:
                # Group close warning labels of the same type and add counters
                for w in warns:
                    wt = getattr(w, "timestamp", None)
                    if not wt:
                        continue
                    txt = str(getattr(w, "label", "") or getattr(w, "warning_type", ""))
                    # Optionally show group name instead of label when grouping is active
                    if group_labels and txt and label_groups:
                        groups = [
                            g for g, members in label_groups.items() if txt in members
                        ]
                        if groups:
                            txt = ",".join(map(str, groups))
                    # Handle grouping and counters
                    if (
                        prev_warning == w.warning_type
                        and prev_time is not None
                        and (wt - prev_time) < timedelta(seconds=1600)
                    ):
                        warning_counter += 1
                        severity_sum += w.severity if hasattr(w, "severity") else 0
                        txt = f"{txt} (!: {severity_sum}) (wngs: {warning_counter})"
                        group_with_prev = True
                    else:
                        group_with_prev = False
                        warning_counter = 1
                        severity_sum = round(
                            w.severity if hasattr(w, "severity") else 0, 2
                        )
                        txt = f"{txt} (!: {severity_sum})"
                        if prev_time is not None and (wt - prev_time) < timedelta(
                            seconds=90 * 60
                        ):
                            marker_y += 1.00
                        else:
                            marker_y = 0.0
                    prev_time = wt
                    prev_warning = w.warning_type
                    if not group_with_prev:
                        grouped_warnings.append((w, wt, txt, marker_y))
                    else:
                        # Update last entry
                        if grouped_warnings:
                            grouped_warnings[-1] = (w, wt, txt, marker_y)

            for w in warns:
                wt = getattr(w, "timestamp", None)
                if not wt:
                    continue
                try:
                    fig.add_vline(
                        x=wt,
                        line_width=2,
                        line_color=color_map.get(w.warning_type, "crimson"),
                        opacity=0.5,
                    )
                except Exception:
                    continue

            # Use pre-grouped warnings with counters
            for gw, wt, txt, marker_y in grouped_warnings:
                try:
                    fig.add_annotation(
                        x=wt,
                        y=marker_y,
                        xref="x",
                        yref="y",
                        xanchor="left",
                        text=txt,
                        showarrow=False,
                        font=dict(size=10, color="white"),
                        bgcolor=color_map.get(gw.warning_type, "crimson"),
                        bordercolor="crimson",
                        borderwidth=0.5,
                        opacity=0.95,
                        textangle=0,
                    )

                except Exception:
                    pass

        # Loudness subplot
        if show_loudness and rows == 2:
            loud_bins = self.loudness_avg_bins(
                start=t0, end=t_end, bin_seconds=bin_seconds
            )
            xs = times
            ys = (
                [float(loud_bins.get(i, np.nan)) for i in range(num_bins)]
                if loud_bins
                else []
            )
            if ys:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name="avg loudness",
                        line=dict(color="#1f77b4"),
                    ),
                    row=2,
                    col=1,
                )
            n, mu, sigma = self.loudness_stats(start=t0, end=t_end)
            eff_sigma = max(float(sigma), float(loud_min_std))
            thr = float(mu) + float(loud_sigma_mult) * eff_sigma
            if times:
                xs2 = [times[0], times[-1]]
                fig.add_trace(
                    go.Scatter(
                        x=xs2,
                        y=[mu, mu],
                        mode="lines",
                        name="mean",
                        line=dict(color="#888", dash="dash"),
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=xs2,
                        y=[thr, thr],
                        mode="lines",
                        name="threshold",
                        line=dict(color="#e67e22", dash="dot"),
                    ),
                    row=2,
                    col=1,
                )
            fig.update_yaxes(title_text="Loudness", row=2, col=1)
            fig.update_xaxes(title_text="time", row=2, col=1, rangebreaks=rangebreaks)

        # Place a single global legend outside the plot area to avoid overlapping subplots/colorbar
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=(loud_h - 25) / (heat_h + loud_h),
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            ),
            margin=dict(l=40, r=60, t=70, b=40),
            height=heat_h + (loud_h if show_loudness else 0) + 100,
            template="plotly_white",
        )
        return fig


# Legacy scaffold removed; DB-centric Visualizer above should be used.
