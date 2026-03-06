import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    'bg': '#0D1117', 'card': '#161B22', 'text': '#E6EDF3',
    'grid': '#30363D', 'muted': '#8B949E',
    'loan_yes': '#00C49F', 'loan_no': '#FF6B6B',
    'accent': '#2196F3', 'warning': '#FF9800',
    'purple': '#9C27B0', 'teal': '#009688', 'success': '#4CAF50',
}

METHOD_COLORS = {
    'Pearson Corr':   '#FF6B6B',
    'Mutual Info':    '#FF9800',
    'RF Importance':  '#00C49F',
    'Permutation':    '#2196F3',
    'Drop-One AUC':   '#9C27B0',
    'LR Coefficient': '#FFD700',
}

FEATURE_VERDICTS = {
    'Income':             ('🔥 Critical',  '#FF4444', 'Dominant predictor across ALL 6 methods. Removing it drops AUC by 6 points. Non-negotiable.'),
    'CCAvg':              ('⚡ Strong',    '#FF9800', 'Consistently 2nd-ranked. High spenders are prime loan candidates.'),
    'Education':          ('⚡ Strong',    '#FF9800', 'Advanced degree holders are 3× more likely to accept. Captures life-stage financial need.'),
    'CD Account':         ('✅ Moderate',  '#00C49F', 'Existing depositors trust the bank. Pearson overstates it due to binary nature.'),
    'Family':             ('✅ Moderate',  '#00C49F', 'Larger families have higher financial needs. Confirmed by permutation importance.'),
    'Mortgage':           ('🔵 Weak',     '#2196F3', 'Slight signal — mortgage holders may need liquidity. Correlation misleadingly low.'),
    'Age':                ('🔵 Weak',     '#2196F3', 'Non-linear effect: mid-career (35-55) peaks. Linear correlation near zero.'),
    'Experience':         ('⚪ Noise',    '#8B949E', 'Almost entirely collinear with Age. Adds no independent information.'),
    'CreditCard':         ('⚪ Noise',    '#8B949E', 'p=0.84 — statistically insignificant. Do NOT use as a targeting signal.'),
    'Securities Account': ('⚪ Noise',    '#8B949E', 'p=0.12 — not significant. Removing it has zero AUC impact.'),
    'Online':             ('⚪ Noise',    '#8B949E', 'p=0.66 — irrelevant to loan intent. Slightly hurts model when over-weighted.'),
}

def method_comparison_grouped(result_df):
    methods = ['Pearson Corr', 'Mutual Info', 'RF Importance', 'Permutation', 'Drop-One AUC', 'LR Coefficient']
    features = result_df.index.tolist()

    fig = go.Figure()
    for method in methods:
        vals = result_df[method].tolist()
        fig.add_trace(go.Bar(
            name=method,
            x=features,
            y=vals,
            marker_color=METHOD_COLORS[method],
            opacity=0.85,
            hovertemplate=f'<b>%{{x}}</b><br>{method}: %{{y:.3f}}<extra></extra>',
        ))
    fig.update_layout(
        barmode='group',
        title=dict(text='6-Method Feature Importance Comparison (Normalized 0–1)',
                   font=dict(size=17, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Feature', gridcolor=COLORS['grid'], tickangle=-20),
        yaxis=dict(title='Normalized Importance Score', gridcolor=COLORS['grid'], range=[0, 1.1]),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.12, x=0),
        height=460,
        margin=dict(t=100, b=60),
    )
    return fig

def composite_score_bar(result_df):
    df = result_df.sort_values('Composite Score', ascending=True).copy()
    colors = []
    for f in df.index:
        verdict = FEATURE_VERDICTS.get(f, ('', '#8B949E', ''))[1]
        colors.append(verdict)

    fig = go.Figure(go.Bar(
        x=df['Composite Score'],
        y=df.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color=COLORS['bg'], width=1)),
        text=[f"  #{int(result_df.loc[f,'Rank'])}  {f}" for f in df.index],
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(color='white', size=12),
        customdata=[[FEATURE_VERDICTS.get(f, ('', '', 'No description'))[2]] for f in df.index],
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<br><br>%{customdata[0]}<extra></extra>',
    ))
    # Legend patches
    for label, color in [('🔥 Critical', '#FF4444'), ('⚡ Strong', '#FF9800'),
                          ('✅ Moderate', '#00C49F'), ('🔵 Weak', '#2196F3'), ('⚪ Noise', '#8B949E')]:
        fig.add_trace(go.Bar(x=[None], y=[None], name=label,
                              marker_color=color, showlegend=True))
    fig.update_layout(
        title=dict(text='Composite Feature Importance Rank (Average of 6 Methods)',
                   font=dict(size=17, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Composite Score (0 = no importance, 1 = critical)', gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.18, x=0),
        height=480,
        margin=dict(t=70, b=100),
        showlegend=True,
    )
    return fig

def radar_chart(result_df):
    methods = ['Pearson Corr', 'Mutual Info', 'RF Importance', 'Permutation', 'Drop-One AUC', 'LR Coefficient']
    top5 = result_df.head(5).index.tolist()
    palette = ['#FF4444', '#FF9800', '#00C49F', '#2196F3', '#9C27B0']

    fig = go.Figure()
    for feature, color in zip(top5, palette):
        vals = result_df.loc[feature, methods].tolist()
        vals += [vals[0]]  # close radar
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=methods + [methods[0]],
            fill='toself',
            name=feature,
            line=dict(color=color, width=2),
            fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in color else color + '26',
            opacity=0.9,
            hovertemplate=f'<b>{feature}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>',
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS['card'],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS['grid'],
                            tickfont=dict(color=COLORS['muted'], size=9)),
            angularaxis=dict(gridcolor=COLORS['grid'], tickfont=dict(color=COLORS['text'], size=11)),
        ),
        title=dict(text='Top 5 Features — Radar Profile Across All Methods',
                   font=dict(size=17, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.12),
        height=480,
        margin=dict(t=80, b=80),
    )
    return fig

def pearson_vs_truth_divergence(result_df):
    """Show where Pearson misleads vs composite truth"""
    df = result_df.copy()
    df['Pearson Rank'] = df['Pearson Corr'].rank(ascending=False).astype(int)
    df['True Rank'] = df['Rank']
    df['Rank Shift'] = df['Pearson Rank'] - df['True Rank']  # + means pearson overrated
    df = df.sort_values('True Rank')

    colors = ['#FF4444' if v > 0 else '#00C49F' if v < 0 else '#8B949E'
              for v in df['Rank Shift']]
    labels = [f"+{v} (Pearson overstates)" if v > 0
              else f"{v} (Pearson understates)" if v < 0
              else "Agrees" for v in df['Rank Shift']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Pearson Corr'],
        name='Pearson Correlation',
        marker_color='#FF6B6B', opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Pearson: %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Composite Score'],
        name='Composite Truth Score',
        mode='lines+markers',
        line=dict(color='#00C49F', width=3),
        marker=dict(size=10, color='#00C49F',
                    line=dict(color='white', width=2)),
        hovertemplate='<b>%{x}</b><br>Composite: %{y:.4f}<extra></extra>',
    ))
    # Annotate biggest divergences
    for feat in df.index:
        shift = df.loc[feat, 'Rank Shift']
        if abs(shift) >= 2:
            fig.add_annotation(
                x=feat,
                y=max(df.loc[feat, 'Pearson Corr'], df.loc[feat, 'Composite Score']) + 0.04,
                text=f"{'▲' if shift > 0 else '▼'}{abs(shift)} ranks",
                font=dict(size=10, color='#FF9800' if shift > 0 else '#2196F3'),
                showarrow=False,
            )
    fig.update_layout(
        title=dict(text='Where Pearson Misleads vs. True Composite Importance',
                   font=dict(size=17, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Feature', gridcolor=COLORS['grid'], tickangle=-15),
        yaxis=dict(title='Normalized Score', gridcolor=COLORS['grid'], range=[0, 1.2]),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
        height=450,
        margin=dict(t=90, b=60),
    )
    return fig

def drop_one_waterfall(raw_df, baseline_auc):
    """Waterfall showing AUC drop when each feature removed"""
    drop = raw_df['Drop-One AUC (raw)'].sort_values(ascending=False)
    drop_pct = (drop * 100).round(4)

    colors = ['#FF4444' if v > 0.005 else '#FF9800' if v > 0.001 else '#30363D'
              for v in drop.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=drop.index,
        y=drop_pct.values,
        marker=dict(color=colors, line=dict(color=COLORS['bg'], width=1)),
        text=[f'-{v:.4f}' if v > 0 else f'+{abs(v):.4f}' for v in drop_pct.values],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11),
        hovertemplate='<b>%{x}</b><br>AUC drops by: %{y:.4f}%<br>'
                      f'AUC with feature: {baseline_auc*100:.3f}%<extra></extra>',
    ))
    fig.add_hline(y=0, line_color=COLORS['muted'], line_width=1)
    fig.update_layout(
        title=dict(text=f'AUC Drop When Feature Removed (Baseline AUC = {baseline_auc:.4f})',
                   font=dict(size=17, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Feature Removed', gridcolor=COLORS['grid'], tickangle=-15),
        yaxis=dict(title='AUC Drop (%)', gridcolor=COLORS['grid']),
        height=420,
        margin=dict(t=70, b=80),
        annotations=[
            dict(text='🔴 Critical drop  🟠 Meaningful  ⬛ Negligible',
                 x=0.5, y=1.08, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=12, color=COLORS['muted']))
        ]
    )
    return fig

def perm_importance_with_error(raw_df):
    """Permutation importance with std error bars"""
    df = raw_df[['Permutation (raw)', 'Perm Std']].sort_values('Permutation (raw)', ascending=False)
    df = df[df['Permutation (raw)'] > -0.01]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Permutation (raw)'],
        error_y=dict(type='data', array=df['Perm Std'].tolist(),
                     color=COLORS['muted'], thickness=2, width=6),
        marker=dict(
            color=df['Permutation (raw)'],
            colorscale=[[0, '#161B22'], [0.3, '#1E4D8C'], [1, '#00C49F']],
            showscale=True,
            colorbar=dict(title='Score', tickfont=dict(color=COLORS['text'])),
        ),
        hovertemplate='<b>%{x}</b><br>Permutation Importance: %{y:.5f}<br>±Std: %{error_y.array:.5f}<extra></extra>',
    ))
    fig.add_hline(y=0, line_color='#FF6B6B', line_dash='dash', line_width=1.5,
                  annotation_text='Zero line (below = noise/harmful)',
                  annotation_font_color='#FF6B6B')
    fig.update_layout(
        title=dict(text='Permutation Importance with Uncertainty (10 repeats)',
                   font=dict(size=17, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Feature', gridcolor=COLORS['grid'], tickangle=-15),
        yaxis=dict(title='Mean Accuracy Drop When Shuffled', gridcolor=COLORS['grid']),
        height=420,
        margin=dict(t=70, b=80),
    )
    return fig

def verdict_summary_table(result_df):
    """Rich verdict dataframe for display"""
    rows = []
    for feat in result_df.index:
        verdict_label, color, desc = FEATURE_VERDICTS.get(feat, ('Unknown', '#8B949E', ''))
        rows.append({
            'Feature': feat,
            'Rank': int(result_df.loc[feat, 'Rank']),
            'Verdict': verdict_label,
            'Composite Score': f"{result_df.loc[feat, 'Composite Score']:.4f}",
            'Pearson': f"{result_df.loc[feat, 'Pearson Corr']:.4f}",
            'RF Importance': f"{result_df.loc[feat, 'RF Importance']:.4f}",
            'Permutation': f"{result_df.loc[feat, 'Permutation']:.4f}",
            'Drop-One AUC': f"{result_df.loc[feat, 'Drop-One AUC']:.4f}",
            'Insight': desc,
        })
    return pd.DataFrame(rows).sort_values('Rank')
