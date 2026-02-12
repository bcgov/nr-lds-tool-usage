"""
lds_usage_dashboard.py

LDS Tool Usage Dashboard - Static HTML Generator

Reads all monthly JSONL files matching *_summary.jsonl and *_detail.jsonl patterns
from the NRS ObjectStore.

Detail logs are joined by run_id to enrich error messages beyond what the summary captures.

"""

import os
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_FILE = r"W:\srm\gss\sandbox\mlabiadh\workspace\20260130_lds_logs\dashboard.html"

# S3-compatible object storage configuration
S3_BUCKET = "gssgeodrive"
S3_PREFIX = "authorizations/new folder|143/lds_tool_logs/"

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_NRS_ENDPOINT"),
    aws_access_key_id=os.getenv("S3_GSS_GEODRIVE_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_GSS_GEODRIVE_SECRET_KEY"),
)

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Developer IDIR(s) to exclude from all stats (test runs)
EXCLUDED_USERS = {'MLABIADH'}

# GIS specialists — everyone else is categorized as "Non-GIS"
GIS_USERS = {'MSEASTWO', 'ALLSHEPH', 'SEPARSON', 'AERASMUS', 'JBUSSE',
             'JFOY', 'CSOSTAD', 'JSANDERS'}

GROUP_GIS = 'GIS Users'
GROUP_NON_GIS = 'Non-GIS Users'

# =============================================================================
# COLOR SCHEME
# =============================================================================
COLORS = {
    'bg_primary': '#1a1a2e',
    'bg_secondary': '#16213e',
    'accent': '#e94560',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'error': '#ef4444',
    'text': '#e2e8f0',
    'text_muted': '#94a3b8',
    'chart': ['#e94560', '#4ade80', '#fbbf24', '#38bdf8', '#a78bfa', '#fb923c']
}

# Stages to skip when extracting the "best" error from detail logs.
# These stages typically echo the root-cause error recorded in earlier stages.
_GENERIC_ERROR_STAGES = {'completion', 'main'}

# =============================================================================
# S3 HELPERS
# =============================================================================
def _list_s3_keys(suffix):
    """List all object keys under S3_PREFIX that end with the given suffix."""
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
    return sorted(keys)


def _read_jsonl_from_s3(key):
    """Download a JSONL file from S3 and return a DataFrame."""
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    body = response["Body"].read()
    return pd.read_json(io.BytesIO(body), lines=True)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load data from all monthly JSONL files in S3."""

    # --- Load summary files ---
    summary_keys = _list_s3_keys("_summary.jsonl")

    if not summary_keys:
        print(f"! No summary JSONL files found under s3://{S3_BUCKET}/{S3_PREFIX}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(summary_keys)} summary file(s)")

    summary_dfs = []
    for key in summary_keys:
        try:
            df_temp = _read_jsonl_from_s3(key)
            filename = key.rsplit("/", 1)[-1]
            print(f"  ✓ Loaded {len(df_temp)} records from {filename}")
            summary_dfs.append(df_temp)
        except Exception as e:
            filename = key.rsplit("/", 1)[-1]
            print(f"  ! Error loading {filename}: {e}")

    if not summary_dfs:
        print("! No summary data loaded")
        return pd.DataFrame(), pd.DataFrame()

    df_summary = pd.concat(summary_dfs, ignore_index=True)
    print(f"✓ Total summary records: {len(df_summary)}")

    df_summary['timestamp_start'] = pd.to_datetime(df_summary['timestamp_start'])
    df_summary['date'] = df_summary['timestamp_start'].dt.date
    df_summary['hour'] = df_summary['timestamp_start'].dt.hour

    # --- Load detail files ---
    detail_keys = _list_s3_keys("_detail.jsonl")
    print(f"\nFound {len(detail_keys)} detail file(s)")

    detail_dfs = []
    for key in detail_keys:
        try:
            df_temp = _read_jsonl_from_s3(key)
            filename = key.rsplit("/", 1)[-1]
            print(f"  ✓ Loaded {len(df_temp)} records from {filename}")
            detail_dfs.append(df_temp)
        except Exception as e:
            filename = key.rsplit("/", 1)[-1]
            print(f"  ! Error loading {filename}: {e}")

    df_detail = pd.concat(detail_dfs, ignore_index=True) if detail_dfs else pd.DataFrame()
    if not df_detail.empty:
        print(f"✓ Total detail records: {len(df_detail)}")

    return df_summary, df_detail


def enrich_errors_from_detail(df_summary, df_detail):
    """
    Join detail-level error info onto the summary dataframe.

    For each run_id, the detail log may contain multiple ERROR-level records
    across different stages (e.g. initialization, ast_execution, completion).
    This function picks the most informative error per run:
      - Prefers errors from specific stages over generic ones (completion/main).
      - Captures the error stage so the dashboard can show *where* the error occurred.
      - Fills in error messages for runs that the summary log missed entirely.

    New columns added to df_summary:
      - detail_error_message : str   – best error message from detail log
      - detail_error_stage   : str   – stage where the error occurred
    """
    if df_detail.empty:
        df_summary['detail_error_message'] = df_summary['error_message']
        df_summary['detail_error_stage'] = None
        return df_summary

    # Filter to ERROR-level records only
    errors = df_detail[df_detail['level'] == 'ERROR'].copy()

    if errors.empty:
        df_summary['detail_error_message'] = df_summary['error_message']
        df_summary['detail_error_stage'] = None
        return df_summary

    # Flag generic stages so we can deprioritise them
    errors['is_generic'] = errors['stage'].isin(_GENERIC_ERROR_STAGES)

    # Sort: specific stages first, then by timestamp so we pick the earliest root cause
    errors = errors.sort_values(['run_id', 'is_generic', 'timestamp'])

    # Take the first (best) error per run_id
    best_errors = errors.groupby('run_id').first().reset_index()

    # Build the enriched error message: clean up common prefixes for readability
    best_errors['detail_error_message'] = best_errors['message'].apply(_clean_error_message)
    best_errors['detail_error_stage'] = best_errors['stage']

    # Merge onto summary
    df_summary = df_summary.merge(
        best_errors[['run_id', 'detail_error_message', 'detail_error_stage']],
        on='run_id',
        how='left'
    )

    # If detail didn't have an error for a run, fall back to the summary's error_message
    mask_no_detail = df_summary['detail_error_message'].isna()
    df_summary.loc[mask_no_detail, 'detail_error_message'] = df_summary.loc[
        mask_no_detail, 'error_message'
    ]

    n_enriched = df_summary['detail_error_message'].notna().sum()
    n_summary_only = df_summary['error_message'].notna().sum()
    n_new = (
        df_summary['detail_error_message'].notna() & df_summary['error_message'].isna()
    ).sum()
    print(f"\n✓ Error enrichment: {n_enriched} runs with errors "
          f"({n_summary_only} from summary, +{n_new} additional from detail logs)")

    return df_summary


def enrich_ast_duration(df_summary, df_detail):
    """
    Compute actual AST processing duration from detail-log timestamps.

    The summary log's duration_seconds and timestamp_end are written BEFORE
    AST starts (by design — so LDS success is captured even if AST takes 40+ min).
    The actual AST duration is only available in the detail log, where the
    ast_execution stage has:
      - A "Starting stage: ast_execution" record (always present for AST runs)
      - An "AST completed successfully" record (present after the fix to
        lands_authorization_v1.py that explicitly logs completion)
      - Or an "AST failed..." ERROR record (captured via the UsageLoggingAdapter
        which forwards WARNING+ level messages)

    This function computes AST duration only for SUCCESSFUL AST runs — i.e.
    runs where the last ast_execution detail record is an INFO-level success
    message, not an ERROR.

    New columns added to df_summary:
      - ast_duration_seconds : float  – AST processing time (NaN if not available)
      - ast_completed        : bool   – True only if AST completed successfully with timing
    """
    df_summary['ast_duration_seconds'] = float('nan')
    df_summary['ast_completed'] = False

    if df_detail.empty:
        return df_summary

    # Filter to ast_execution stage records
    ast_records = df_detail[df_detail['stage'] == 'ast_execution'].copy()
    if ast_records.empty:
        return df_summary

    ast_records['timestamp'] = pd.to_datetime(ast_records['timestamp'])

    # For each run_id, check if AST completed successfully
    grouped = ast_records.groupby('run_id')

    durations = {}
    completed_runs = set()
    for run_id, grp in grouped:
        grp_sorted = grp.sort_values('timestamp')
        if len(grp_sorted) < 2:
            continue  # Only "Starting stage" — no end record, skip

        last_record = grp_sorted.iloc[-1]

        # Only count successful AST completions (not failures)
        if last_record['level'] == 'ERROR':
            continue

        start = grp_sorted.iloc[0]['timestamp']
        end = last_record['timestamp']
        dur = (end - start).total_seconds()
        durations[run_id] = dur
        completed_runs.add(run_id)

    if durations:
        dur_series = pd.Series(durations, name='ast_duration_seconds')
        df_summary = df_summary.set_index('run_id')
        df_summary['ast_duration_seconds'] = dur_series
        df_summary['ast_completed'] = df_summary.index.isin(completed_runs)
        df_summary = df_summary.reset_index()

        n_timed = len(durations)
        n_ast = (df_summary['ast'] == True).sum()
        print(f"\n✓ AST duration enrichment: {n_timed}/{n_ast} AST runs completed successfully with timing")
    else:
        print(f"\n! No successful AST runs with timing data found in detail logs")

    return df_summary


def _clean_error_message(msg):
    """
    Normalize error messages for grouping.

    Strips common prefixes like 'Unexpected Exception: ' and file-specific
    prefixes like 'Error in LDS run for row N: File XXXXXXX - ' so that
    identical root causes group together in the chart.
    """
    import re
    if not isinstance(msg, str):
        return msg

    # Remove 'Unexpected Exception: ' prefix
    msg = re.sub(r'^Unexpected Exception:\s*', '', msg)

    # Remove 'Error in LDS run for row N: File XXXXXXX - ' prefix
    msg = re.sub(r'^Error in LDS run for row \d+:\s*File \S+\s*-\s*', '', msg)

    # Remove 'AST failed with unexpected error: ' prefix (keep the actual ArcGIS error)
    msg = re.sub(r'^AST failed with unexpected error:\s*', '', msg)

    # Truncate very long messages (e.g. ArcGIS stack traces) at first newline
    if '\n' in msg:
        msg = msg.split('\n')[0].strip()

    # Truncate to 120 chars for chart readability
    if len(msg) > 120:
        msg = msg[:117] + '...'

    return msg.strip()


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def clean_username(username):
    """Remove IDIR\\ prefix from username and normalize to uppercase."""
    if isinstance(username, str):
        return username.replace('IDIR\\', '').replace('IDIR/', '').upper()
    return username

def assign_user_group(clean_user):
    """Assign user to GIS or Non-GIS group."""
    if isinstance(clean_user, str) and clean_user.upper() in GIS_USERS:
        return GROUP_GIS
    return GROUP_NON_GIS

def calculate_metrics(df):
    """Calculate all metrics from dataframe."""
    total = len(df)

    # Duration stats by AST
    # Without AST: all non-AST runs
    ast_false = df[df['ast'] == False]['duration_seconds']
    # With AST: use ast_duration_seconds (from detail logs) where available,
    # otherwise fall back to summary duration_seconds for completed AST runs.
    # ast_duration_seconds captures actual AST processing time computed from
    # the detail log stage timestamps (start → end of ast_execution stage).
    if 'ast_duration_seconds' in df.columns:
        ast_true = df.loc[
            (df['ast'] == True) & (df['ast_completed'] == True),
            'ast_duration_seconds'
        ].dropna()
    else:
        ast_true = df[(df['ast'] == True) & (df['status'] == 'success')]['duration_seconds']

    # Date range
    date_min = df['timestamp_start'].min().strftime('%Y-%m-%d')
    date_max = df['timestamp_start'].max().strftime('%Y-%m-%d')

    # Error counting now uses detail_error_message (enriched) if available
    error_col = 'detail_error_message' if 'detail_error_message' in df.columns else 'error_message'
    has_error = df[error_col].astype(str).str.strip().str.len() > 0
    has_error = has_error & (df[error_col].notna())

    # Group counts
    gis_runs = len(df[df['user_group'] == GROUP_GIS])
    non_gis_runs = len(df[df['user_group'] == GROUP_NON_GIS])

    # P90 duration (all runs)
    p90_duration = df['duration_seconds'].quantile(0.90) if total > 0 else 0

    # AST success rate (among runs that requested AST)
    ast_requested = len(df[df['ast'] == True])
    if 'ast_completed' in df.columns:
        ast_succeeded = df.loc[df['ast'] == True, 'ast_completed'].sum()
    else:
        ast_succeeded = len(df[(df['ast'] == True) & (df['status'] == 'success')])
    ast_success_rate = (ast_succeeded / ast_requested * 100) if ast_requested > 0 else 0

    return {
        'total_runs': total,
        'unique_machines': df['machine'].nunique(),
        'unique_users': df['clean_user'].nunique(),
        'gis_users': df.loc[df['user_group'] == GROUP_GIS, 'clean_user'].nunique(),
        'non_gis_users': df.loc[df['user_group'] == GROUP_NON_GIS, 'clean_user'].nunique(),
        'gis_runs': gis_runs,
        'non_gis_runs': non_gis_runs,
        # Median pipeline duration (without AST)
        'median_duration_without_ast': ast_false.median() if len(ast_false) > 0 else 0,
        # Median AST processing time (completed AST runs only, from detail logs)
        'median_duration_with_ast': ast_true.median() if len(ast_true) > 0 else 0,
        'ast_completed_count': int(ast_succeeded),
        'ast_requested_count': ast_requested,
        'ast_success_rate': ast_success_rate,
        # P90 duration (all runs)
        'p90_duration': p90_duration,
        'peak_hour': df['hour'].mode().iloc[0] if len(df['hour'].mode()) > 0 else 0,
        'busiest_day': df.groupby('date').size().idxmax(),
        'success_rate': len(df[df['status'] == 'success']) / total * 100,
        'error_rate': has_error.sum() / total * 100,
        'warning_rate': len(df[df['warning_count'] > 0]) / total * 100,
        'error_types': df.loc[has_error, error_col].nunique(),
        'date_from': date_min,
        'date_to': date_max,
    }

# =============================================================================
# CHART CREATION
# =============================================================================
def get_chart_layout(title="", height=300):
    """Return consistent chart layout."""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': COLORS['text'], 'family': 'system-ui, -apple-system, sans-serif', 'size': 12},
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 60},
        'title': {'text': title, 'font': {'size': 14, 'color': COLORS['text_muted']}},
        'xaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'zerolinecolor': 'rgba(255,255,255,0.1)'},
        'yaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'zerolinecolor': 'rgba(255,255,255,0.1)'},
        'height': height,
        'autosize': True,
    }

def create_daily_trend(df):
    daily = df.groupby(['date', 'user_group']).size().reset_index(name='runs')
    fig = go.Figure()
    for group, color in [(GROUP_GIS, COLORS['chart'][3]), (GROUP_NON_GIS, COLORS['chart'][5])]:
        grp = daily[daily['user_group'] == group]
        fig.add_trace(go.Scatter(
            x=grp['date'], y=grp['runs'], mode='lines+markers',
            name=group, line=dict(color=color), marker=dict(color=color)
        ))
    fig.update_layout(**get_chart_layout('Daily Run Trend'))
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    return fig

def create_user_distribution_gis(df):
    """Bar chart of runs by GIS users."""
    gis_df = df[df['user_group'] == GROUP_GIS]
    if gis_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No GIS user runs", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**get_chart_layout('Runs by GIS Users', height=320))
        return fig
    user_counts = gis_df['clean_user'].value_counts().head(10).reset_index()
    user_counts.columns = ['user', 'count']
    fig = px.bar(user_counts, x='count', y='user', orientation='h',
                 color_discrete_sequence=[COLORS['chart'][3]])
    fig.update_layout(**get_chart_layout('Runs by GIS Users', height=320))
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_user_distribution_non_gis(df):
    """Bar chart of runs by Non-GIS users."""
    non_gis_df = df[df['user_group'] == GROUP_NON_GIS]
    if non_gis_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Non-GIS user runs", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**get_chart_layout('Runs by Non-GIS Users', height=320))
        return fig
    user_counts = non_gis_df['clean_user'].value_counts().head(10).reset_index()
    user_counts.columns = ['user', 'count']
    fig = px.bar(user_counts, x='count', y='user', orientation='h',
                 color_discrete_sequence=[COLORS['chart'][5]])
    fig.update_layout(**get_chart_layout('Runs by Non-GIS Users', height=320))
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_region_distribution(df):
    region_counts = df.groupby(['ast_region', 'user_group']).size().reset_index(name='count')
    color_map = {GROUP_GIS: COLORS['chart'][3], GROUP_NON_GIS: COLORS['chart'][5]}
    fig = px.bar(region_counts, x='count', y='ast_region', orientation='h',
                 color='user_group', color_discrete_map=color_map, barmode='stack')
    fig.update_layout(**get_chart_layout('Runs by Region'))
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, title_text=''),
    )
    return fig

def create_failure_rate_trend(df):
    """Weekly failure-rate (%) trend with regional context.

    Layers (bottom to top):
      1. Shaded band showing the overall average failure rate for reference.
      2. Faint regional lines so you can spot which regions need attention.
      3. Bold overall failure-rate line (errors / total runs per week).
      4. 3-week moving average trend line to smooth out weekly noise.

    Failure rate is computed as:
        errors_in_week / total_runs_in_week * 100

    Uses the same region → color mapping as the Errors by Region pie chart.
    """
    import numpy as np

    error_col = 'detail_error_message' if 'detail_error_message' in df.columns else 'error_message'

    df_copy = df.copy()
    df_copy['week'] = df_copy['timestamp_start'].dt.to_period('W').apply(lambda r: r.start_time)
    df_copy['has_error'] = (
        df_copy[error_col].notna()
        & (df_copy[error_col].astype(str).str.strip().str.len() > 0)
    )

    # --- Overall weekly failure rate ---
    weekly = df_copy.groupby('week').agg(
        total=('run_id', 'size'),
        errors=('has_error', 'sum'),
    ).reset_index()
    weekly['failure_rate'] = weekly['errors'] / weekly['total'] * 100
    weekly = weekly.sort_values('week')

    if weekly.empty or weekly['total'].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**get_chart_layout('Weekly Failure Rate Trend'))
        return fig

    # Overall average failure rate (for the reference band)
    overall_avg = weekly['errors'].sum() / weekly['total'].sum() * 100

    # 3-week centred moving average (min_periods=1 so edges aren't NaN)
    weekly['ma'] = weekly['failure_rate'].rolling(3, center=True, min_periods=1).mean()

    # --- Per-region weekly failure rate ---
    region_weekly = df_copy.groupby(['week', 'ast_region']).agg(
        total=('run_id', 'size'),
        errors=('has_error', 'sum'),
    ).reset_index()
    region_weekly['failure_rate'] = region_weekly['errors'] / region_weekly['total'] * 100

    # Build stable color map: sort regions by total errors descending (matches pie)
    region_order = (
        df_copy.loc[df_copy['has_error'], 'ast_region']
        .value_counts()
        .index.tolist()
    )
    # Include regions that may have zero errors too (for completeness)
    all_regions = df_copy['ast_region'].unique()
    for r in all_regions:
        if r not in region_order:
            region_order.append(r)

    region_color_map = {
        region: COLORS['chart'][i % len(COLORS['chart'])]
        for i, region in enumerate(region_order)
    }

    fig = go.Figure()

    # --- Layer 1: Overall average reference band ---
    fig.add_hrect(
        y0=max(overall_avg - 2, 0), y1=overall_avg + 2,
        fillcolor=COLORS['text_muted'], opacity=0.08,
        line_width=0,
    )
    fig.add_hline(
        y=overall_avg,
        line_dash='dot', line_color=COLORS['text_muted'], line_width=1,
        annotation_text=f"Avg {overall_avg:.1f}%",
        annotation_position='top left',
        annotation_font=dict(size=11, color=COLORS['text_muted']),
    )

    # --- Layer 2: Faint regional lines ---
    for region in region_order:
        grp = region_weekly[region_weekly['ast_region'] == region].sort_values('week')
        if grp.empty:
            continue
        fig.add_trace(go.Scatter(
            x=grp['week'], y=grp['failure_rate'],
            mode='lines',
            name=region,
            line=dict(color=region_color_map[region], width=1.7),
            opacity=0.5,
            hovertemplate=(
                f"<b>{region}</b><br>"
                "Week: %{x|%b %d}<br>"
                "Failure rate: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))

    # --- Layer 3: Bold overall failure rate ---
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly['failure_rate'],
        mode='lines+markers',
        name='Overall',
        line=dict(color=COLORS['text'], width=3),
        marker=dict(color=COLORS['text'], size=6),
        hovertemplate=(
            "<b>Overall</b><br>"
            "Week: %{x|%b %d}<br>"
            "Failure rate: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # --- Layer 4: Moving average trend line ---
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly['ma'],
        mode='lines',
        name='3-wk trend',
        line=dict(color=COLORS['accent'], width=2.5, dash='dash'),
        hovertemplate=(
            "<b>3-week moving avg</b><br>"
            "Week: %{x|%b %d}<br>"
            "Trend: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(**get_chart_layout('Weekly Failure Rate Trend', height=350))
    fig.update_layout(
        yaxis_title='Failure Rate (%)',
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)',
            rangemode='tozero',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    return fig

def create_status_distribution(df):
    status = df['status'].value_counts().reset_index()
    status.columns = ['status', 'count']
    colors = [COLORS['success'] if s == 'success' else COLORS['error'] for s in status['status']]
    fig = px.pie(status, values='count', names='status', color_discrete_sequence=colors, hole=0.4)
    fig.update_layout(**get_chart_layout('Status Distribution', height=320))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_error_messages(df):
    """
    Bar chart of common error messages, enriched from detail logs.

    Uses 'detail_error_message' (joined from detail JSONL) which provides:
      - More specific root-cause messages than the summary log
      - Errors from runs the summary missed (e.g. AST failures on otherwise 
        'success' runs)
      - Cleaned/normalized messages so identical root causes group together
    
    Bars are color-coded by the stage where the error occurred.
    Hover shows GIS vs Non-GIS breakdown.
    """
    error_col = 'detail_error_message' if 'detail_error_message' in df.columns else 'error_message'
    stage_col = 'detail_error_stage' if 'detail_error_stage' in df.columns else None

    # Filter to rows with non-empty error messages
    mask = df[error_col].notna() & (df[error_col].astype(str).str.strip().str.len() > 0)
    cols = [error_col, 'user_group']
    if stage_col:
        cols.append(stage_col)
    error_df = df.loc[mask, cols].copy()

    if len(error_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No errors recorded", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**get_chart_layout('Common Error Messages'))
        return fig

    # Count errors by message (and stage if available)
    if stage_col:
        # Group by message, pick the most common stage for each message
        grouped = error_df.groupby(error_col).agg(
            count=(error_col, 'size'),
            stage=(stage_col, lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'),
            gis_count=('user_group', lambda x: (x == GROUP_GIS).sum()),
            non_gis_count=('user_group', lambda x: (x == GROUP_NON_GIS).sum()),
        ).reset_index()
        grouped.columns = ['message', 'count', 'stage', 'gis_count', 'non_gis_count']
        grouped = grouped.sort_values('count', ascending=False).head(8)

        # Color map for stages
        stage_colors = {
            'initialization': COLORS['chart'][0],     # red
            'input_validation': COLORS['chart'][2],    # yellow
            'workspace_creation': COLORS['chart'][5],  # orange
            'ast_execution': COLORS['chart'][4],       # purple
            'tenure_info': COLORS['chart'][3],         # blue
            'admin_overlap': COLORS['chart'][1],       # green
            'batch_run': COLORS['text_muted'],
        }

        # Create figure with stage-colored bars
        fig = go.Figure()
        for _, row in grouped.iterrows():
            color = stage_colors.get(row['stage'], COLORS['error'])
            fig.add_trace(go.Bar(
                y=[row['message']],
                x=[row['count']],
                orientation='h',
                marker_color=color,
                name=row['stage'],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['message']}</b><br>"
                    f"Stage: {row['stage']}<br>"
                    f"Count: {row['count']}<br>"
                    f"GIS: {row['gis_count']} | Non-GIS: {row['non_gis_count']}"
                    f"<extra></extra>"
                ),
                text=[f"{row['count']}  [{row['stage']}]"],
                textposition='outside',
                textfont=dict(size=11),
            ))

        fig.update_layout(**get_chart_layout('Common Error Messages (from Detail Logs)', height=380))
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin={'l': 250, 'r': 100, 't': 50, 'b': 60},
        )
    else:
        # Fallback: no stage info, simple bar chart (original behavior)
        errors = error_df[error_col].value_counts().head(8).reset_index()
        errors.columns = ['message', 'count']
        fig = px.bar(errors, x='count', y='message', orientation='h',
                     color_discrete_sequence=[COLORS['error']])
        fig.update_layout(**get_chart_layout('Common Error Messages', height=380))

    return fig

def create_error_by_region(df):
    """
    Pie chart showing error distribution by region.
    """
    error_col = 'detail_error_message' if 'detail_error_message' in df.columns else 'error_message'

    # Filter to rows with non-empty error messages
    mask = df[error_col].notna() & (df[error_col].astype(str).str.strip().str.len() > 0)
    error_df = df.loc[mask].copy()

    if len(error_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No errors recorded", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**get_chart_layout('Errors by Region', height=320))
        return fig

    region_errors = error_df['ast_region'].value_counts().reset_index()
    region_errors.columns = ['region', 'count']

    fig = px.pie(region_errors, values='count', names='region',
                 color_discrete_sequence=COLORS['chart'], hole=0.4)
    fig.update_layout(**get_chart_layout('Errors by Region', height=320))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_user_group_split(df):
    """Donut chart showing % of runs by GIS vs Non-GIS users."""
    group_counts = df['user_group'].value_counts().reset_index()
    group_counts.columns = ['group', 'count']
    color_map = {GROUP_GIS: COLORS['chart'][3], GROUP_NON_GIS: COLORS['chart'][5]}
    colors = [color_map.get(g, COLORS['text_muted']) for g in group_counts['group']]
    fig = px.pie(group_counts, values='count', names='group',
                 color_discrete_sequence=colors, hole=0.4)
    fig.update_layout(**get_chart_layout('Runs by User Group', height=320))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_feature_adoption(df):
    total = len(df)
    features = pd.DataFrame({
        'feature': ['Layer Input', 'Inset Map', 'Prov Ref Map', 'AST', 'Replace Hyper', 'Legal Desc'],
        'rate': [
            df['layer_input_provided'].sum() / total * 100,
            df['inset_map'].sum() / total * 100,
            df['prov_ref_map'].sum() / total * 100,
            df['ast'].sum() / total * 100,
            df['replace_hyper'].sum() / total * 100,
            df['input_legal_desc_provided'].sum() / total * 100,
        ]
    })
    fig = px.bar(features, x='rate', y='feature', orientation='h', color_discrete_sequence=[COLORS['chart'][4]],
                 text=[f"{v:.1f}%" for v in features['rate']])
    fig.update_layout(**get_chart_layout('Feature Usage Rates', height=350))
    fig.update_xaxes(title='Adoption %', range=[0, 100])
    fig.update_traces(textposition='outside')
    fig.update_layout(margin={'l': 100, 'r': 60, 't': 50, 'b': 60})
    return fig

def create_prov_ref_by_region(df):
    stats = df.groupby('ast_region').agg(total=('run_id', 'count'), prov_ref=('prov_ref_map', 'sum')).reset_index()
    stats['usage_pct'] = stats['prov_ref'] / stats['total'] * 100
    fig = px.bar(stats, x='ast_region', y='usage_pct', color_discrete_sequence=[COLORS['accent']],
                 text=[f"{v:.0f}%" for v in stats['usage_pct']])
    fig.update_layout(**get_chart_layout('Provincial Ref Map Usage by Region', height=350))
    fig.update_yaxes(title='Usage %', range=[0, 100])
    fig.update_traces(textposition='outside')
    return fig

# =============================================================================
# HTML GENERATION
# =============================================================================
def generate_html(df, metrics):
    """Generate complete HTML dashboard."""
    
    # Create all charts
    charts = {
        'daily_trend': create_daily_trend(df).to_html(full_html=False, include_plotlyjs=False),
        'user_dist_gis': create_user_distribution_gis(df).to_html(full_html=False, include_plotlyjs=False),
        'user_dist_non_gis': create_user_distribution_non_gis(df).to_html(full_html=False, include_plotlyjs=False),
        'region_dist': create_region_distribution(df).to_html(full_html=False, include_plotlyjs=False),
        'failure_rate_trend': create_failure_rate_trend(df).to_html(full_html=False, include_plotlyjs=False),
        'status_dist': create_status_distribution(df).to_html(full_html=False, include_plotlyjs=False),
        'error_msgs': create_error_messages(df).to_html(full_html=False, include_plotlyjs=False),
        'error_region': create_error_by_region(df).to_html(full_html=False, include_plotlyjs=False),
        'user_group_split': create_user_group_split(df).to_html(full_html=False, include_plotlyjs=False),
        'feature_adoption': create_feature_adoption(df).to_html(full_html=False, include_plotlyjs=False),
        'prov_ref_region': create_prov_ref_by_region(df).to_html(full_html=False, include_plotlyjs=False),
    }
    
    # Format duration values
    def format_duration(seconds):
        if seconds >= 60:
            mins = seconds / 60
            return f"{mins:.1f}m"
        return f"{seconds:.0f}s"
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LDS Tool Usage Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, {COLORS['bg_primary']} 0%, #0f0f1a 50%, {COLORS['bg_primary']} 100%);
            color: {COLORS['text']};
            min-height: 100vh;
            padding: 32px;
        }}
        
        header {{
            margin-bottom: 40px;
            padding-bottom: 24px;
            border-bottom: 1px solid rgba(233, 69, 96, 0.3);
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            background: {COLORS['success']};
            border-radius: 50%;
            margin-right: 12px;
            box-shadow: 0 0 12px {COLORS['success']};
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .subtitle {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 3px;
            color: {COLORS['text_muted']};
            font-family: monospace;
        }}
        
        h1 {{
            font-size: 42px;
            font-weight: 700;
            letter-spacing: -1px;
            margin-top: 8px;
        }}
        
        section {{
            margin-bottom: 48px;
        }}
        
        .section-header {{
            font-size: 20px;
            text-transform: uppercase;
            letter-spacing: 4px;
            color: {COLORS['accent']};
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid {COLORS['accent']};
            font-family: monospace;
            font-weight: 700;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {COLORS['bg_secondary']} 0%, {COLORS['bg_primary']} 100%);
            border: 1px solid rgba(233, 69, 96, 0.2);
            border-radius: 4px;
            padding: 20px;
        }}
        
        .metric-card .label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: {COLORS['text_muted']};
            margin-bottom: 8px;
            font-family: monospace;
        }}
        
        .metric-card .value {{
            font-size: 32px;
            font-weight: 700;
            color: {COLORS['text']};
            margin-bottom: 4px;
        }}
        
        .metric-card .card-subtitle {{
            font-size: 12px;
            color: {COLORS['text_muted']};
            font-family: monospace;
            letter-spacing: 0;
            text-transform: none;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }}
        
        .charts-grid-3 {{
            grid-template-columns: repeat(3, 1fr);
        }}
        
        .chart-container {{
            background: linear-gradient(180deg, {COLORS['bg_secondary']} 0%, {COLORS['bg_primary']} 100%);
            border: 1px solid rgba(233, 69, 96, 0.15);
            border-radius: 4px;
            padding: 16px;
            overflow: hidden;
        }}
        
        .chart-container .js-plotly-plot {{
            width: 100% !important;
        }}
        
        .chart-container .plotly {{
            width: 100% !important;
        }}
        
        footer {{
            text-align: center;
            padding-top: 24px;
            border-top: 1px solid rgba(233, 69, 96, 0.2);
        }}
        
        footer p {{
            font-size: 11px;
            color: {COLORS['text_muted']};
            font-family: monospace;
            letter-spacing: 1px;
        }}
        
        @media (max-width: 1200px) {{
            .charts-grid-3 {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 16px;
            }}
            h1 {{
                font-size: 28px;
            }}
            .charts-grid, .charts-grid-3 {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <span class="status-indicator"></span>
            <span class="subtitle">Data Period: {metrics['date_from']} to {metrics['date_to']}</span>
        </div>
        <h1>LDS Tool Usage Dashboard</h1>
    </header>
    
    <!-- USAGE VOLUME -->
    <section>
        <h2 class="section-header">Usage Volume</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Runs</div>
                <div class="value">{metrics['total_runs']}</div>
                <div class="card-subtitle">All records</div>
            </div>
            <div class="metric-card">
                <div class="label">Unique Users</div>
                <div class="value">{metrics['unique_users']}</div>
                <div class="card-subtitle">{metrics['gis_users']} GIS &bull; {metrics['non_gis_users']} Non-GIS</div>
            </div>
            <div class="metric-card">
                <div class="label">GIS Runs</div>
                <div class="value">{metrics['gis_runs']}</div>
                <div class="card-subtitle">GIS specialist runs</div>
            </div>
            <div class="metric-card">
                <div class="label">Non-GIS Runs</div>
                <div class="value">{metrics['non_gis_runs']}</div>
                <div class="card-subtitle">Non-GIS user runs</div>
            </div>
        </div>
        <div class="charts-grid charts-grid-3">
            <div class="chart-container">{charts['daily_trend']}</div>
            <div class="chart-container">{charts['region_dist']}</div>
            <div class="chart-container">{charts['user_group_split']}</div>
        </div>
        <div class="charts-grid" style="margin-top: 16px;">
            <div class="chart-container">{charts['user_dist_gis']}</div>
            <div class="chart-container">{charts['user_dist_non_gis']}</div>
        </div>
    </section>
    
    <!-- PERFORMANCE & RELIABILITY -->
    <section>
        <h2 class="section-header">Performance & Reliability</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Median LDS Time</div>
                <div class="value">{format_duration(metrics['median_duration_without_ast'])}</div>
                <div class="card-subtitle">Runs without AST</div>
            </div>
            <div class="metric-card">
                <div class="label">Median AST Time</div>
                <div class="value">{format_duration(metrics['median_duration_with_ast'])}</div>
                <div class="card-subtitle">Runs with AST</div>
            </div>
            <div class="metric-card">
                <div class="label">Success Rate</div>
                <div class="value">{metrics['success_rate']:.1f}%</div>
                <div class="card-subtitle">Completed runs</div>
            </div>
            <div class="metric-card">
                <div class="label">Warning Rate</div>
                <div class="value">{metrics['warning_rate']:.1f}%</div>
                <div class="card-subtitle">Runs with warnings</div>
            </div>
            <div class="metric-card">
                <div class="label">Error Types</div>
                <div class="value">{metrics['error_types']}</div>
                <div class="card-subtitle">Unique errors</div>
            </div>
        </div>
        <div class="charts-grid" style="margin-top: 16px;">
            <div class="chart-container" style="grid-column: span 2;">{charts['failure_rate_trend']}</div>
        </div>
        <div class="charts-grid" style="margin-top: 16px;">
            <div class="chart-container">{charts['status_dist']}</div>
            <div class="chart-container">{charts['error_region']}</div>
        </div>
        <div class="charts-grid" style="margin-top: 16px;">
            <div class="chart-container" style="grid-column: span 2;">{charts['error_msgs']}</div>
        </div>
    </section>
    
    <!-- FEATURE ADOPTION -->
    <section>
        <h2 class="section-header">Feature Adoption</h2>
        <div class="charts-grid">
            <div class="chart-container">{charts['feature_adoption']}</div>
            <div class="chart-container">{charts['prov_ref_region']}</div>
        </div>
    </section>
    
    <footer>
        <p>Tool Usage Analytics &bull; Data from monthly JSONL logs (summary + detail)</p>
    </footer>
    
    <script>
        // Make all Plotly charts responsive
        window.addEventListener('resize', function() {{
            document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
                Plotly.Plots.resize(plot);
            }});
        }});
        
        // Initial resize to fit containers
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 100);
        }});
    </script>
</body>
</html>'''
    
    return html

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("LDS Tool Usage Dashboard - HTML Generator")
    print("="*60)
    
    # Load summary and detail data
    df_summary, df_detail = load_data()
    
    # Clean usernames: strip IDIR prefix, normalize to uppercase
    df_summary['clean_user'] = df_summary['user_os'].apply(clean_username)
    
    # Exclude developer test runs
    before = len(df_summary)
    df_summary = df_summary[~df_summary['clean_user'].isin(EXCLUDED_USERS)].copy()
    excluded = before - len(df_summary)
    if excluded > 0:
        print(f"\n✓ Excluded {excluded} developer test runs ({', '.join(EXCLUDED_USERS)})")
    
    # Assign user groups (GIS vs Non-GIS)
    df_summary['user_group'] = df_summary['clean_user'].apply(assign_user_group)
    gis_n = (df_summary['user_group'] == GROUP_GIS).sum()
    non_gis_n = (df_summary['user_group'] == GROUP_NON_GIS).sum()
    print(f"✓ User groups: {gis_n} GIS runs, {non_gis_n} Non-GIS runs")
    
    # Enrich error messages from detail logs
    df = enrich_errors_from_detail(df_summary, df_detail)
    
    # Enrich AST duration from detail logs (stage-level timestamps)
    df = enrich_ast_duration(df, df_detail)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Generate HTML
    html_content = generate_html(df, metrics)
    
    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Generated {OUTPUT_FILE}")
    print("="*60 + "\n")