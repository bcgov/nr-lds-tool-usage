"""
LDS Tool Usage Dashboard - Static HTML Generator
Generates a standalone HTML file with interactive Plotly charts.

Reads all monthly JSONL files matching *_summary.jsonl and *_detail.jsonl patterns.
Detail logs are joined by run_id to enrich error messages beyond what the summary captures.

"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURATION
# =============================================================================
PATH = r"\\objectstore2.nrs.bcgov\GSS_Share\authorizations\logs\lds_tool_logs"
OUTPUT_FILE = r"W:\srm\gss\sandbox\mlabiadh\workspace\20260130_lds_logs\dashboard.html"

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Developer IDIR(s) to exclude from all stats (test runs)
EXCLUDED_USERS = {'MLABIADH'}

# GIS specialists — everyone else is categorized as "Non-GIS"
GIS_USERS = {'MSEASTWO', 'ALLSHEPH', 'SEPARSON', 'AERASMUS', 'JBUSSE'}

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
# DATA LOADING
# =============================================================================
def load_data():
    """Load data from all monthly JSONL files."""
    import glob

    # --- Load summary files ---
    summary_files = glob.glob(os.path.join(PATH, '*_summary.jsonl'))

    if not summary_files:
        print(f"! No summary JSONL files found in {PATH}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(summary_files)} summary file(s)")

    summary_dfs = []
    for jsonl_file in sorted(summary_files):
        try:
            df_temp = pd.read_json(jsonl_file, lines=True)
            filename = os.path.basename(jsonl_file)
            print(f"  ✓ Loaded {len(df_temp)} records from {filename}")
            summary_dfs.append(df_temp)
        except Exception as e:
            print(f"  ! Error loading {os.path.basename(jsonl_file)}: {e}")

    if not summary_dfs:
        print("! No summary data loaded")
        return pd.DataFrame(), pd.DataFrame()

    df_summary = pd.concat(summary_dfs, ignore_index=True)
    print(f"✓ Total summary records: {len(df_summary)}")

    df_summary['timestamp_start'] = pd.to_datetime(df_summary['timestamp_start'])
    df_summary['date'] = df_summary['timestamp_start'].dt.date
    df_summary['hour'] = df_summary['timestamp_start'].dt.hour

    # --- Load detail files ---
    detail_files = glob.glob(os.path.join(PATH, '*_detail.jsonl'))
    print(f"\nFound {len(detail_files)} detail file(s)")

    detail_dfs = []
    for jsonl_file in sorted(detail_files):
        try:
            df_temp = pd.read_json(jsonl_file, lines=True)
            filename = os.path.basename(jsonl_file)
            print(f"  ✓ Loaded {len(df_temp)} records from {filename}")
            detail_dfs.append(df_temp)
        except Exception as e:
            print(f"  ! Error loading {os.path.basename(jsonl_file)}: {e}")

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
    ast_true = df[df['ast'] == True]['duration_seconds']
    ast_false = df[df['ast'] == False]['duration_seconds']

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

    return {
        'total_runs': total,
        'unique_machines': df['machine'].nunique(),
        'unique_users': df['clean_user'].nunique(),
        'gis_users': df.loc[df['user_group'] == GROUP_GIS, 'clean_user'].nunique(),
        'non_gis_users': df.loc[df['user_group'] == GROUP_NON_GIS, 'clean_user'].nunique(),
        'gis_runs': gis_runs,
        'non_gis_runs': non_gis_runs,
        # Duration with AST (median only)
        'median_duration_with_ast': ast_true.median() if len(ast_true) > 0 else 0,
        # Duration without AST (median only)
        'median_duration_without_ast': ast_false.median() if len(ast_false) > 0 else 0,
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

def create_duration_by_region(df):
    """Bar chart showing median duration by region, split by AST selection."""
    # Calculate median duration per region for AST and non-AST
    regions = df['ast_region'].unique()
    
    data = []
    for region in regions:
        region_data = df[df['ast_region'] == region]
        
        # Without AST
        no_ast = region_data[region_data['ast'] == False]['duration_seconds']
        if len(no_ast) > 0:
            data.append({'region': region, 'type': 'Without AST', 'median_duration': no_ast.median()})
        
        # With AST
        with_ast = region_data[region_data['ast'] == True]['duration_seconds']
        if len(with_ast) > 0:
            data.append({'region': region, 'type': 'With AST', 'median_duration': with_ast.median()})
    
    plot_df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # Add bars for Without AST
    no_ast_df = plot_df[plot_df['type'] == 'Without AST']
    fig.add_trace(go.Bar(
        name='Without AST',
        y=no_ast_df['region'],
        x=no_ast_df['median_duration'],
        orientation='h',
        marker_color=COLORS['chart'][1],
        text=[f"{v:.0f}s" for v in no_ast_df['median_duration']],
        textposition='outside'
    ))
    
    # Add bars for With AST
    with_ast_df = plot_df[plot_df['type'] == 'With AST']
    fig.add_trace(go.Bar(
        name='With AST',
        y=with_ast_df['region'],
        x=with_ast_df['median_duration'],
        orientation='h',
        marker_color=COLORS['chart'][0],
        text=[f"{v:.0f}s" for v in with_ast_df['median_duration']],
        textposition='outside'
    ))
    
    fig.update_layout(**get_chart_layout('Median Duration by Region', height=320))
    fig.update_layout(
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis_title='Median Duration (seconds)'
    )
    return fig

def create_peak_hours(df):
    hours = df['hour'].value_counts().sort_index().reset_index()
    hours.columns = ['hour', 'runs']
    hours['label'] = hours['hour'].apply(lambda h: f"{h}:00")
    fig = px.bar(hours, x='label', y='runs', color_discrete_sequence=[COLORS['chart'][3]])
    fig.update_layout(**get_chart_layout('Peak Usage Times'))
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

def create_error_by_stage(df):
    """
    Bar chart showing error count by pipeline stage.
    
    Only available when detail logs have been loaded (detail_error_stage column).
    This gives a quick overview of which stages are most error-prone.
    """
    stage_col = 'detail_error_stage'
    if stage_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Detail logs required for stage breakdown",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(**get_chart_layout('Errors by Pipeline Stage'))
        return fig

    mask = df[stage_col].notna()
    if mask.sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No stage-level errors recorded",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**get_chart_layout('Errors by Pipeline Stage'))
        return fig

    stage_counts = df.loc[mask, stage_col].value_counts().reset_index()
    stage_counts.columns = ['stage', 'count']

    # Color map for stages
    stage_colors = {
        'initialization': COLORS['chart'][0],
        'input_validation': COLORS['chart'][2],
        'workspace_creation': COLORS['chart'][5],
        'ast_execution': COLORS['chart'][4],
        'tenure_info': COLORS['chart'][3],
        'admin_overlap': COLORS['chart'][1],
        'batch_run': COLORS['text_muted'],
    }
    colors = [stage_colors.get(s, COLORS['error']) for s in stage_counts['stage']]

    fig = go.Figure(go.Bar(
        y=stage_counts['stage'],
        x=stage_counts['count'],
        orientation='h',
        marker_color=colors,
        text=stage_counts['count'],
        textposition='outside',
    ))
    fig.update_layout(**get_chart_layout('Errors by Pipeline Stage', height=320))
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
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
        'duration_region': create_duration_by_region(df).to_html(full_html=False, include_plotlyjs=False),
        'peak_hours': create_peak_hours(df).to_html(full_html=False, include_plotlyjs=False),
        'status_dist': create_status_distribution(df).to_html(full_html=False, include_plotlyjs=False),
        'error_msgs': create_error_messages(df).to_html(full_html=False, include_plotlyjs=False),
        'error_stage': create_error_by_stage(df).to_html(full_html=False, include_plotlyjs=False),
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
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 3px;
            color: {COLORS['accent']};
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid {COLORS['accent']};
            font-family: monospace;
            font-weight: 600;
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
    
    <!-- PERFORMANCE -->
    <section>
        <h2 class="section-header">Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Median (No AST)</div>
                <div class="value">{format_duration(metrics['median_duration_without_ast'])}</div>
                <div class="card-subtitle">Without AST selection</div>
            </div>
            <div class="metric-card">
                <div class="label">Median (With AST)</div>
                <div class="value">{format_duration(metrics['median_duration_with_ast'])}</div>
                <div class="card-subtitle">With AST selection</div>
            </div>
        </div>
        <div class="charts-grid">
            <div class="chart-container">{charts['duration_region']}</div>
            <div class="chart-container">{charts['peak_hours']}</div>
        </div>
    </section>
    
    <!-- RELIABILITY -->
    <section>
        <h2 class="section-header">Reliability</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Success Rate</div>
                <div class="value">{metrics['success_rate']:.1f}%</div>
                <div class="card-subtitle">Completed runs</div>
            </div>
            <div class="metric-card">
                <div class="label">Error Rate</div>
                <div class="value">{metrics['error_rate']:.1f}%</div>
                <div class="card-subtitle">Runs with errors</div>
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
        <div class="charts-grid">
            <div class="chart-container">{charts['status_dist']}</div>
            <div class="chart-container">{charts['error_stage']}</div>
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
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Generate HTML
    html_content = generate_html(df, metrics)
    
    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Generated {OUTPUT_FILE}")
    print("="*60 + "\n")