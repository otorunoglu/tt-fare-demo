import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
import pandas as pd
from typing import List, Tuple, Optional
from matplotlib.axes import Axes  # type: ignore

def load_and_visualize_metamodel_data(
    json_file_path,
    feeding_windows: Optional[List[Tuple[time, time]]] = None,
    loudness_anomaly_threshold: float = 0.2,
    show: bool = True,
    output_path: Optional[str] = None,
    minimal: bool = False,
    gap_break_minutes: int = 10,
    confidence_mode: str = "clip",
):
    """
    Load metamodel JSON data and create comprehensive visualizations
    """
    # Load the data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract analysis results from either single-task or range output
    results = []
    if 'analysis_results' in data:
        # Single-task payload
        results = data['analysis_results'] or []
    elif 'per_task' in data:
        # Range payload
        per_task = data.get('per_task', [])
        include_details = bool(data.get('include_details', False))
        # If detailed results exist, flatten them
        if include_details:
            for entry in per_task:
                task_results = entry.get('analysis_results', []) or []
                for r in task_results:
                    results.append(r)
        else:
            # No details provided; synthesize one point per task using summary and recorded time
            for entry in per_task:
                ts = None
                try:
                    rd = entry.get('recorded_date')
                    rt = entry.get('recorded_time') or '00:00:00'
                    if rd:
                        ts = datetime.strptime(f"{rd} {rt}", "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = None
                # Create a minimal datum for visualization purposes
                summary = entry.get('summary', {})
                # Pick the highest alert type if any
                alert_breakdown = summary.get('alert_breakdown', {})
                primary_alert = None
                if alert_breakdown:
                    primary_alert = max(alert_breakdown, key=lambda k: alert_breakdown.get(k, 0))
                    if alert_breakdown.get(primary_alert, 0) == 0:
                        primary_alert = None
                results.append({
                    'timestamp': ts.isoformat() if isinstance(ts, datetime) else None,
                    'primary_alert': primary_alert,
                    'confidence': None,
                    'contributing_factors': {},
                    'label_distribution': {},
                    'loudness_metrics': {},
                    'time_context': {}
                })
    
    # Convert to pandas DataFrame for easier manipulation
    df_data = []
    for result in results:
        # Robust timestamp parsing
        ts_val = result.get('timestamp')
        if ts_val:
            try:
                timestamp = datetime.fromisoformat(ts_val)
            except Exception:
                try:
                    timestamp = datetime.strptime(ts_val, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    timestamp = None
        else:
            timestamp = None
        
        # Extract basic info
        row = {
            'timestamp': timestamp,
            'primary_alert': result.get('primary_alert'),
            'confidence': result.get('confidence', 0) if result.get('confidence') is not None else 0,
            'num_contributing_factors': len(result.get('contributing_factors', {}) or {}),
            'hour': timestamp.hour if isinstance(timestamp, datetime) else None,
            'minute': timestamp.minute if isinstance(timestamp, datetime) else None,
            'second': timestamp.second if isinstance(timestamp, datetime) else None
        }
        
        # Extract label distribution
        for label, rate in (result.get('label_distribution') or {}).items():
            row[f'label_{label}_rate'] = rate
        
        # Extract loudness metrics
        for window_type in ['immediate', 'stable', 'trend']:
            metrics = (result.get('loudness_metrics') or {}).get(window_type)
            if metrics:
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[f'{window_type}_{metric}'] = value
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Drop rows without timestamps for time-based plots
    df = df.dropna(subset=['timestamp'])

    # Create visualizations
    create_comprehensive_plots(
        df,
        data,
        feeding_windows=feeding_windows or [],
        loudness_anomaly_threshold=loudness_anomaly_threshold,
        show=show,
        output_path=output_path,
        minimal=minimal,
        gap_break_minutes=gap_break_minutes,
        confidence_mode=confidence_mode,
    )

def create_comprehensive_plots(
    df,
    raw_data,
    feeding_windows: List[Tuple[time, time]],
    loudness_anomaly_threshold: float,
    show: bool,
    output_path: Optional[str],
    minimal: bool,
    gap_break_minutes: int,
    confidence_mode: str,
):
    """Create multiple plots to understand the data"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    if minimal:
        fig = plt.figure(figsize=(18, 10))
    else:
        fig = plt.figure(figsize=(20, 16))
    
    # Prepare optional subplot handles for static analysis
    ax2: Optional[Axes] = None
    ax4: Optional[Axes] = None
    ax5: Optional[Axes] = None
    ax6: Optional[Axes] = None

    # Summary info
    # Header fields for single-task vs range
    is_range = 'per_task' in raw_data
    if not is_range:
        stable_id = raw_data.get('stable_id')
        task_id = raw_data.get('task_id')
        total_alerts = raw_data.get('summary', {}).get('total_alerts', 0)
        title = f'MetaModel Analysis for {stable_id} - Task {task_id}\n' \
                f'Total Alerts: {total_alerts} | Status: {raw_data.get("summary", {}).get("overall_status", "").upper()}'
    else:
        project_id = raw_data.get('project_id')
        date_from = raw_data.get('date_from')
        date_until = raw_data.get('date_until')
        total_alerts = raw_data.get('summary', {}).get('total_alerts', 0)
        title = f'MetaModel Range Analysis for Project {project_id} ({date_from}..{date_until})\n' \
                f'Total Alerts: {total_alerts} | Status: {raw_data.get("summary", {}).get("overall_status", "").upper()}'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Determine layout differences if we only have synthesized per_task points (no details)
    has_details = 'analysis_results' in raw_data or (raw_data.get('include_details') and 'per_task' in raw_data)

    # Confidence normalization handling
    if has_details and not df.empty and 'confidence' in df.columns:
        max_conf = df['confidence'].max()
        if confidence_mode == 'clip':
            if max_conf > 1:
                print(f"[warn] Clipping confidence values >1 (max observed {max_conf:.2f})")
            df['confidence_for_plot'] = df['confidence'].clip(0, 1)
        elif confidence_mode == 'scale':
            if max_conf > 0:
                df['confidence_for_plot'] = df['confidence'] / max_conf
                if max_conf > 1:
                    print(f"[info] Scaled confidence by max {max_conf:.2f} to fit [0,1]")
            else:
                df['confidence_for_plot'] = df['confidence']
        else:  # raw
            df['confidence_for_plot'] = df['confidence']

    # 1. Timeline of Alerts and Confidence (or total alerts if no details)
    ax1 = plt.subplot(3 if minimal else 4, 1 if minimal else 2, 1)
    if has_details and not df.empty:
        ax1.plot(df['timestamp'], df.get('confidence_for_plot', df['confidence']), color='blue', alpha=0.7, linewidth=1, label='Confidence')
    elif 'per_task' in raw_data:
        import pandas as _pd
        pts = []
        for entry in raw_data.get('per_task', []):
            rd = entry.get('recorded_date')
            rt = entry.get('recorded_time') or '00:00:00'
            try:
                ts = datetime.strptime(f"{rd} {rt}", "%Y-%m-%d %H:%M:%S") if rd else None
            except Exception:
                ts = None
            summary = entry.get('summary', {})
            pts.append({
                'timestamp': ts,
                'total_alerts': summary.get('total_alerts', 0),
                **{k: summary.get('alert_breakdown', {}).get(k, 0) for k in ['confirmed_burst','possible_burst','kick_concern','trend_concern']}
            })
        per_task_df = _pd.DataFrame(pts).dropna(subset=['timestamp']).sort_values('timestamp')
        if not per_task_df.empty:
            ax1.plot(per_task_df['timestamp'], per_task_df['total_alerts'], color='darkred', label='Total Alerts', linewidth=1.5)
            if len(per_task_df) > 3:
                per_task_df['rolling'] = per_task_df['total_alerts'].rolling(window=3, min_periods=1).mean()
                ax1.plot(per_task_df['timestamp'], per_task_df['rolling'], color='orange', linestyle='--', label='Rolling Mean (3)')
    
    
    # Highlight alerts (only if we have detailed windows)
    if has_details and not df.empty:
        alert_mask = df['primary_alert'].notna()
        if alert_mask.any():
            alert_df = df[alert_mask]
            alert_colors = {
                'kick_concern': 'red',
                'confirmed_burst': 'darkred',
                'possible_burst': 'orange',
                'trend_concern': 'purple',
                'night_disturbance': 'navy'
            }
            for alert_type in alert_df['primary_alert'].unique():
                mask = alert_df['primary_alert'] == alert_type
                if mask.any():
                    subset = alert_df[mask]
                    ax1.scatter(subset['timestamp'], subset['confidence'],
                                color=alert_colors.get(alert_type, 'gray'),
                                s=100, alpha=0.8, label=f'{alert_type} ({len(subset)})',
                                marker='o', edgecolors='black')
    
    ax1.set_ylabel('Confidence' if has_details else 'Total Alerts')
    ax1.set_title('Alerts Timeline' if has_details else 'Per-Task Alert Totals')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Label Distribution Over Time or Alert Breakdown Over Time (range summary mode)
    if not minimal:
        ax2 = plt.subplot(4, 2, 2)
    if not minimal and ax2 is not None and has_details:
        label_cols = [col for col in df.columns if col.startswith('label_') and col.endswith('_rate')]
        if label_cols:
            for col in label_cols:
                label_name = col.replace('label_', '').replace('_rate', '')
                if df[col].max() > 0:
                    ax2.plot(df['timestamp'], df[col], label=label_name, marker='o', markersize=3, alpha=0.8)
            ax2.set_ylabel('Label Rate')
            ax2.set_title('Activity Labels Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No label activity detected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Activity Labels Over Time')
    elif not minimal and ax2 is not None:
        import pandas as _pd
        pts2: List[dict] = []
        for entry in raw_data.get('per_task', []):
            rd = entry.get('recorded_date')
            rt = entry.get('recorded_time') or '00:00:00'
            try:
                ts = datetime.strptime(f"{rd} {rt}", "%Y-%m-%d %H:%M:%S") if rd else None
            except Exception:
                ts = None
            ab = entry.get('summary', {}).get('alert_breakdown', {})
            pts2.append({'timestamp': ts, **ab})
        per_task_alerts = _pd.DataFrame(pts2).dropna(subset=['timestamp']).sort_values('timestamp')
        if not per_task_alerts.empty:
            for col in [c for c in per_task_alerts.columns if c != 'timestamp']:
                if per_task_alerts[col].sum() > 0:
                    ax2.plot(per_task_alerts['timestamp'], per_task_alerts[col], marker='o', linewidth=1, label=col)
            ax2.set_ylabel('Alert Count')
            ax2.set_title('Alert Breakdown Per Task')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No alerts in range', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Alert Breakdown Per Task')
    
    # 3. Loudness Patterns (skip if no detailed windows)
    ax3_index = 2 if minimal else 3
    ax3 = plt.subplot(3 if minimal else 4, 1 if minimal else 2, ax3_index)
    
    if has_details:
        loudness_cols = ['immediate_loudness_mean', 'stable_loudness_mean', 'trend_loudness']
        colors = ['red', 'blue', 'green']

        def _segment_timeseries(ts_series, val_series):
            if ts_series.empty:
                return []
            segments = []
            current_t = [ts_series.iloc[0]]
            current_v = [val_series.iloc[0]]
            for i in range(1, len(ts_series)):
                gap = (ts_series.iloc[i] - ts_series.iloc[i - 1]).total_seconds() / 60.0
                if gap > gap_break_minutes:
                    segments.append((current_t, current_v))
                    current_t = [ts_series.iloc[i]]
                    current_v = [val_series.iloc[i]]
                else:
                    current_t.append(ts_series.iloc[i])
                    current_v.append(val_series.iloc[i])
            segments.append((current_t, current_v))
            return segments

        for i, col in enumerate(loudness_cols):
            if col in df.columns:
                segments = _segment_timeseries(df['timestamp'], df[col])
                for seg_t, seg_v in segments:
                    ax3.plot(seg_t, seg_v, color=colors[i], alpha=0.75, linewidth=2,
                             label=col.replace('_', ' ').title() if seg_t == segments[0][0] else None)

        # Loudness anomaly detection: immediate above stable by threshold
        if 'immediate_loudness_mean' in df.columns and 'stable_loudness_mean' in df.columns:
            diff = df['immediate_loudness_mean'] - df['stable_loudness_mean']
            anomaly_mask = diff > loudness_anomaly_threshold
            if anomaly_mask.any():
                ax3.scatter(df[anomaly_mask]['timestamp'], df[anomaly_mask]['immediate_loudness_mean'],
                            marker='*', s=120, color='gold', edgecolors='black', label=f'Anomaly (Δ>{loudness_anomaly_threshold})')
        ax3.set_ylabel('Loudness')
        ax3.set_title('Multi-Scale Loudness Patterns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No loudness metrics (details disabled)', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Multi-Scale Loudness (N/A)')
    
    # 4. Burstiness Analysis (only if details)
    if not minimal:
        ax4 = plt.subplot(4, 2, 4)
    if not minimal and ax4 is not None and has_details and 'immediate_burstiness' in df.columns:
        burstiness = df['immediate_burstiness'].fillna(0)
        ax4.plot(df['timestamp'], burstiness, color='orange', alpha=0.7, linewidth=2)
        high_burst = burstiness > 2.0
        if high_burst.any():
            ax4.scatter(df[high_burst]['timestamp'], burstiness[high_burst],
                        color='red', s=50, alpha=0.8, label=f'High Burst (>{2.0})')
            ax4.legend()
        ax4.set_ylabel('Burstiness Factor')
        ax4.set_title('Activity Burstiness Over Time')
    if not minimal and ax4 is not None:
        if not (has_details and 'immediate_burstiness' in df.columns):
            ax4.text(0.5, 0.5, 'Burstiness not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Burstiness (N/A)')
        ax4.grid(True, alpha=0.3)
    
    # 5. Hourly Activity Heatmap
    if not minimal:
        ax5 = plt.subplot(4, 2, 5)
    
    # Create activity heatmap by minute
    if not minimal:
        df['time_minutes'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    
    # Aggregate activity by minute
    if not minimal:
        activity_by_minute = df.groupby('time_minutes').agg({
            'confidence': 'max',
            'num_contributing_factors': 'max'
        }).reset_index()
    else:
        import pandas as _pd
        activity_by_minute = _pd.DataFrame()
    
    if not minimal and ax5 is not None and not activity_by_minute.empty:
        # Convert back to hours for display
        activity_by_minute['hour_decimal'] = activity_by_minute['time_minutes'] / 60
        
        scatter = ax5.scatter(activity_by_minute['hour_decimal'], 
                            [1] * len(activity_by_minute),
                            c=activity_by_minute['confidence'], 
                            s=activity_by_minute['num_contributing_factors'] * 50 + 20,
                            cmap='Reds', alpha=0.7)
        
        plt.colorbar(scatter, ax=ax5, label='Confidence')
        ax5.set_xlabel('Hour of Day')
        ax5.set_ylabel('')
        ax5.set_title('Activity Intensity by Hour (size=factors, color=confidence)')
        ax5.set_xlim(df['timestamp'].dt.hour.min() - 0.1, df['timestamp'].dt.hour.max() + 0.1)
    
    # 6. Alert Summary Bar Chart
    if not minimal:
        ax6 = plt.subplot(4, 2, 6)
    
    if not minimal:
        alert_breakdown = raw_data.get('summary', {}).get('alert_breakdown', {})
        alert_types = list(alert_breakdown.keys())
        alert_counts = list(alert_breakdown.values())
    else:
        alert_types = []  # for type checker
        alert_counts = []
    
    bars = None  # type: ignore[assignment]
    if not minimal and ax6 is not None:
        colors_bar = ['red', 'orange', 'yellow', 'purple']
        bars = ax6.bar(alert_types, alert_counts, color=colors_bar[:len(alert_types)], alpha=0.7)
    
    # Add count labels on bars
    if not minimal and ax6 is not None and bars is not None:
        for bar, count in zip(bars, alert_counts):
                if count > 0:
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(count), ha='center', va='bottom', fontweight='bold')
    
    if not minimal and ax6 is not None:
        ax6.set_ylabel('Number of Alerts')
        ax6.set_title('Alert Type Breakdown')
        ax6.set_xticklabels(alert_types, rotation=45, ha='right')
    
    # 7. Detailed Event Timeline (only with details)
    ax7 = plt.subplot(3 if minimal else 4, 1, 3 if minimal else 4)
    
    # Create event timeline showing all events
    event_types = []
    event_times = []
    event_colors = []
    event_sizes = []
    
    color_map = {
        'horse_kick': 'red',
        'neigh': 'blue', 
        'rattle': 'orange',
        'door': 'green',
        'bang': 'purple'
    }
    
    if has_details:
        for _, row in df.iterrows():
            for col in df.columns:
                if col.startswith('label_') and col.endswith('_rate') and row[col] > 0:
                    label_name = col.replace('label_', '').replace('_rate', '')
                    event_types.append(label_name)
                    event_times.append(row['timestamp'])
                    event_colors.append(color_map.get(label_name, 'gray'))
                    event_sizes.append(row[col] * 200 + 20)  # Scale by rate
    
    if has_details and event_types:
        # Create categorical y-positions
        unique_types = list(set(event_types))
        y_positions = [unique_types.index(et) for et in event_types]
        
        ax7.scatter(event_times, y_positions, c=event_colors, s=event_sizes, alpha=0.7)
        ax7.set_yticks(range(len(unique_types)))
        ax7.set_yticklabels(unique_types)
        ax7.set_xlabel('Time')
        ax7.set_title('Detailed Event Timeline (size indicates intensity)')
    else:
        ax7.text(0.5, 0.5, 'No events detected', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Detailed Event Timeline')
    
    ax7.grid(True, alpha=0.3)

    # --- Feeding window shading (applied to relevant time-based axes) ---
    if feeding_windows and not df.empty:
        # Determine unique dates in dataset
        unique_dates = sorted({ts.date() for ts in df['timestamp']})
        shaded_once = {id(ax1): False, id(ax3): False, id(ax7): False}
        axes_to_shade = [ax1, ax3, ax7]
        for day in unique_dates:
            for start_t, end_t in feeding_windows:
                # Handle windows that may cross midnight
                start_dt = datetime.combine(day, start_t)
                end_dt = datetime.combine(day, end_t)
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)
                for ax in axes_to_shade:
                    label = 'Feeding Window' if not shaded_once[id(ax)] else None
                    start_num = float(mdates.date2num(start_dt))
                    end_num = float(mdates.date2num(end_dt))
                    ax.axvspan(start_num, end_num, color='yellow', alpha=0.12, label=label)
                    shaded_once[id(ax)] = True
        # Add legend entry if not already present
        for ax in axes_to_shade:
            handles, labels_existing = ax.get_legend_handles_labels()
            if 'Feeding Window' in labels_existing:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis for all subplots
    axes_for_time = [ax1, ax3, ax7] if minimal else [a for a in [ax1, ax2, ax3, ax4, ax7] if a is not None]
    for ax in axes_for_time:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    if output_path:
        try:
            fig.savefig(output_path, dpi=150)
            print(f"Saved composite figure to {output_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Print summary statistics
    print_summary_stats(df, raw_data, feeding_windows)

def print_summary_stats(df, raw_data, feeding_windows: List[Tuple[time, time]]):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("METAMODEL ANALYSIS SUMMARY")
    print("="*60)
    
    is_range = 'per_task' in raw_data
    if not is_range:
        print(f"Stable ID: {raw_data.get('stable_id')}")
        print(f"Task ID: {raw_data.get('task_id')}")
        print(f"Analysis Period: {raw_data.get('summary', {}).get('analysis_period_minutes', 'n/a')} minutes")
        print(f"Total Windows Analyzed: {raw_data.get('total_windows_analyzed', 'n/a')}")
        print(f"Overall Status: {raw_data.get('summary', {}).get('overall_status', '').upper()}")
    else:
        print(f"Project ID: {raw_data.get('project_id')}")
        print(f"Date Range: {raw_data.get('date_from')} .. {raw_data.get('date_until')}")
        print(f"Tasks Analyzed: {raw_data.get('tasks_analyzed')}")
        print(f"Windows Analyzed Total: {raw_data.get('windows_analyzed_total')}")
        print(f"Overall Status: {raw_data.get('summary', {}).get('overall_status', '').upper()}")
    
    print(f"\nAlert Breakdown:")
    for alert_type, count in (raw_data.get('summary', {}).get('alert_breakdown', {}) or {}).items():
        if count > 0:
            print(f"  {alert_type}: {count}")
    
    print(f"\nTime Range:")
    if not df.empty:
        print(f"  Start: {df['timestamp'].min()}")
        print(f"  End: {df['timestamp'].max()}")
    else:
        print("  No timestamped data available")
    
    print(f"\nActivity Summary:")
    label_cols = [col for col in df.columns if col.startswith('label_') and col.endswith('_rate')]
    for col in label_cols:
        label_name = col.replace('label_', '').replace('_rate', '')
        max_rate = df[col].max()
        total_events = (df[col] > 0).sum()
        if max_rate > 0:
            print(f"  {label_name}: {total_events} windows with activity (max rate: {max_rate:.2f})")
    
    print(f"\nLoudness Statistics:")
    if 'immediate_loudness_mean' in df.columns:
        print(f"  Immediate loudness: {df['immediate_loudness_mean'].mean():.3f} ± {df['immediate_loudness_mean'].std():.3f}")
    if 'stable_loudness_mean' in df.columns:
        print(f"  Stable loudness: {df['stable_loudness_mean'].mean():.3f} ± {df['stable_loudness_mean'].std():.3f}")
    
    if 'immediate_burstiness' in df.columns:
        max_burst = df['immediate_burstiness'].max()
        high_burst_count = (df['immediate_burstiness'] > 2.0).sum()
        print(f"  Max burstiness: {max_burst:.1f}")
        print(f"  High burstiness windows: {high_burst_count}")

    # Feeding window stats
    if feeding_windows and not df.empty:
        print("\nFeeding Window Analysis:")
        unique_dates = sorted({ts.date() for ts in df['timestamp']})
        inside_counts = 0
        outside_counts = 0
        inside_alerts = 0
        outside_alerts = 0
        for _, row in df.iterrows():
            ts = row['timestamp']
            inside = False
            for day in unique_dates:
                for start_t, end_t in feeding_windows:
                    start_dt = datetime.combine(day, start_t)
                    end_dt = datetime.combine(day, end_t)
                    if end_dt <= start_dt:
                        end_dt += timedelta(days=1)
                    if start_dt <= ts < end_dt:
                        inside = True
                        break
                if inside:
                    break
            if inside:
                inside_counts += 1
                if row.get('primary_alert'):
                    inside_alerts += 1
            else:
                outside_counts += 1
                if row.get('primary_alert'):
                    outside_alerts += 1
        total = inside_counts + outside_counts
        if total > 0:
            print(f"  Windows inside feeding windows: {inside_counts} ({inside_counts/total:.1%})")
            print(f"  Windows outside feeding windows: {outside_counts} ({outside_counts/total:.1%})")
        if inside_counts > 0:
            print(f"  Alerts inside feeding windows: {inside_alerts} ({inside_alerts/max(1, inside_counts):.1%} of inside windows)")
        if outside_counts > 0:
            print(f"  Alerts outside feeding windows: {outside_alerts} ({outside_alerts/max(1, outside_counts):.1%} of outside windows)")
        if 'immediate_loudness_mean' in df.columns:
            # Loudness inside/outside
            inside_mask = []
            for ts in df['timestamp']:
                inside = False
                for day in unique_dates:
                    for start_t, end_t in feeding_windows:
                        start_dt = datetime.combine(day, start_t)
                        end_dt = datetime.combine(day, end_t)
                        if end_dt <= start_dt:
                            end_dt += timedelta(days=1)
                        if start_dt <= ts < end_dt:
                            inside = True
                            break
                    if inside:
                        break
                inside_mask.append(inside)
            inside_series = df.loc[inside_mask, 'immediate_loudness_mean']
            outside_series = df.loc[[not i for i in inside_mask], 'immediate_loudness_mean']
            if not inside_series.empty and not outside_series.empty:
                print(f"  Avg immediate loudness inside feeding: {inside_series.mean():.3f}")
                print(f"  Avg immediate loudness outside feeding: {outside_series.mean():.3f}")

def _parse_time_window(spec: str) -> Tuple[time, time]:
    """Parse HH:MM-HH:MM into (time,time)."""
    try:
        start_s, end_s = spec.split('-')
        st = datetime.strptime(start_s.strip(), '%H:%M').time()
        et = datetime.strptime(end_s.strip(), '%H:%M').time()
        return st, et
    except Exception as e:
        raise ValueError(f"Invalid feeding window format '{spec}'. Use HH:MM-HH:MM") from e

def main():
    """Main function to run the visualization with optional feeding windows and anomaly threshold"""
    import sys
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Visualize MetaModel output JSON with optional feeding windows.')
    parser.add_argument('json_file', nargs='?', default='2025_08_31_stable01.json', help='JSON file inside tools/visualize_metamodel directory')
    parser.add_argument('--feeding-window', action='append', default=[], help='Feeding window time range HH:MM-HH:MM (can repeat).')
    parser.add_argument('--loudness-anomaly-threshold', type=float, default=0.2, help='Threshold difference (immediate - stable) to flag loudness anomaly.')
    parser.add_argument('--no-show', action='store_true', help='Do not display interactive window (useful for headless runs).')
    parser.add_argument('--save-fig', default=None, help='Path to save the composite figure PNG.')
    parser.add_argument('--minimal', default=True, action='store_true', help='Show only alerts timeline, loudness, and detailed event timeline.')
    parser.add_argument('--gap-break-minutes', type=int, default=10, help='Gap (minutes) after which to break loudness lines.')
    parser.add_argument('--confidence-mode', choices=['raw','clip','scale'], default='clip', help='How to handle confidence values >1.')
    args = parser.parse_args()

    rel_path = "tools/visualize_metamodel"
    path = os.path.join(rel_path, args.json_file)

    feeding_windows: List[Tuple[time, time]] = []
    for spec in args.feeding_window:
        feeding_windows.append(_parse_time_window(spec))

    try:
        load_and_visualize_metamodel_data(
            path,
            feeding_windows=feeding_windows,
            loudness_anomaly_threshold=args.loudness_anomaly_threshold,
            show=not args.no_show,
            output_path=args.save_fig,
            minimal=args.minimal,
            gap_break_minutes=args.gap_break_minutes,
            confidence_mode=args.confidence_mode,
        )
    except FileNotFoundError:
        print(f"Error: Could not find file '{path}'")
        print("Usage: python visualize_metamodel.py <json_file> [--feeding-window HH:MM-HH:MM]")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()