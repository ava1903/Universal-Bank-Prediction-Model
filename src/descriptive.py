import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    'primary': '#1E3A5F',
    'accent': '#2196F3',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'purple': '#9C27B0',
    'teal': '#009688',
    'bg': '#0D1117',
    'card': '#161B22',
    'text': '#E6EDF3',
    'loan_yes': '#00C49F',
    'loan_no': '#FF6B6B',
}

def loan_distribution_donut(df):
    counts = df['Personal Loan'].value_counts()
    labels = ['Not Interested', 'Interested']
    values = [counts.get(0, 0), counts.get(1, 0)]
    pct = [v / sum(values) * 100 for v in values]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=[COLORS['loan_no'], COLORS['loan_yes']],
                    line=dict(color='#0D1117', width=3)),
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
        pull=[0, 0.05],
    ))
    fig.update_layout(
        title=dict(text='Personal Loan Acceptance Distribution', font=dict(size=18, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        legend=dict(orientation='h', yanchor='bottom', y=-0.1),
        annotations=[dict(text=f'<b>{values[1]:,}</b><br>Accepted', x=0.5, y=0.5,
                          font=dict(size=16, color=COLORS['loan_yes']), showarrow=False)],
        margin=dict(t=60, b=40, l=20, r=20), height=400,
    )
    return fig

def age_income_scatter(df):
    fig = px.scatter(
        df, x='Age', y='Income', color='Personal Loan',
        color_discrete_map={0: COLORS['loan_no'], 1: COLORS['loan_yes']},
        size='CCAvg', size_max=18,
        hover_data={'Age': True, 'Income': True, 'CCAvg': True, 'Education_Label': True,
                    'Personal Loan': False},
        labels={'Personal Loan': 'Loan Status'},
        title='Age vs Income — Sized by CC Spending',
        opacity=0.7,
    )
    fig.update_layout(
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(gridcolor='#30363D', title='Age'),
        yaxis=dict(gridcolor='#30363D', title='Annual Income ($000)'),
        legend=dict(title='Loan', bgcolor='rgba(0,0,0,0)'),
        title=dict(font=dict(size=16, color=COLORS['text'])),
        height=420,
    )
    return fig

def correlation_heatmap(df):
    num_df = df.select_dtypes(include='number').drop(columns=['Personal Loan'], errors='ignore')
    corr = num_df.corrwith(df['Personal Loan']).sort_values(ascending=False)
    full_corr = num_df.assign(**{'Personal Loan': df['Personal Loan']}).corr()

    fig = go.Figure(data=go.Heatmap(
        z=full_corr.values,
        x=full_corr.columns.tolist(),
        y=full_corr.columns.tolist(),
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(full_corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} × %{y}: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Feature Correlation Matrix', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(tickangle=-45),
        height=500,
        margin=dict(t=60, b=100),
    )
    return fig

def feature_distributions(df):
    features = ['Age', 'Income', 'CCAvg', 'Mortgage', 'Experience', 'Family']
    fig = make_subplots(rows=2, cols=3, subplot_titles=features,
                        vertical_spacing=0.15, horizontal_spacing=0.08)
    for i, feat in enumerate(features):
        row, col = divmod(i, 3)
        for loan_val, name, color in [(0, 'No Loan', COLORS['loan_no']), (1, 'Loan', COLORS['loan_yes'])]:
            subset = df[df['Personal Loan'] == loan_val][feat]
            fig.add_trace(go.Histogram(
                x=subset, name=name, opacity=0.75,
                marker_color=color, nbinsx=25,
                showlegend=(i == 0),
                hovertemplate=f'<b>{feat}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>',
            ), row=row+1, col=col+1)
    fig.update_layout(
        barmode='overlay',
        title=dict(text='Feature Distributions by Loan Status', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        height=520,
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    for ax in fig.layout:
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig.layout[ax].update(gridcolor='#30363D', zerolinecolor='#30363D')
    return fig

def categorical_breakdown(df):
    cats = {
        'Education': df['Education_Label'].value_counts(),
        'Family Size': df['Family'].value_counts().sort_index(),
        'Online Banking': df['Online'].map({0: 'No', 1: 'Yes'}).value_counts(),
        'Credit Card': df['CreditCard'].map({0: 'No', 1: 'Yes'}).value_counts(),
        'Securities Acct': df['Securities Account'].map({0: 'No', 1: 'Yes'}).value_counts(),
        'CD Account': df['CD Account'].map({0: 'No', 1: 'Yes'}).value_counts(),
    }
    palette = [COLORS['accent'], COLORS['success'], COLORS['warning'],
               COLORS['purple'], COLORS['teal'], COLORS['danger']]
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=list(cats.keys()),
                        specs=[[{'type': 'domain'}]*3, [{'type': 'domain'}]*3])
    for i, (title, series) in enumerate(cats.items()):
        row, col = divmod(i, 3)
        n = len(series)
        colors = palette[:n]
        fig.add_trace(go.Pie(
            labels=series.index.astype(str).tolist(),
            values=series.values.tolist(),
            hole=0.5,
            marker=dict(colors=colors, line=dict(color='#0D1117', width=2)),
            textinfo='percent', textfont=dict(size=11),
            showlegend=False,
            hovertemplate='<b>%{label}</b>: %{value:,} (%{percent})<extra></extra>',
        ), row=row+1, col=col+1)
    fig.update_layout(
        title=dict(text='Categorical Feature Breakdown', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=500,
        margin=dict(t=80, b=20),
    )
    return fig

def income_band_loan_rate(df):
    tbl = df.groupby('Income_Band', observed=True)['Personal Loan'].agg(['mean', 'sum', 'count']).reset_index()
    tbl.columns = ['Income Band', 'Loan Rate', 'Accepted', 'Total']
    tbl['Loan Rate %'] = (tbl['Loan Rate'] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tbl['Income Band'].astype(str), y=tbl['Total'],
        name='Total Customers', marker_color='#30363D',
        hovertemplate='<b>%{x}</b><br>Total: %{y:,}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        x=tbl['Income Band'].astype(str), y=tbl['Accepted'],
        name='Accepted Loan', marker_color=COLORS['loan_yes'],
        hovertemplate='<b>%{x}</b><br>Accepted: %{y:,}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=tbl['Income Band'].astype(str), y=tbl['Loan Rate %'],
        name='Acceptance Rate %', yaxis='y2',
        line=dict(color=COLORS['warning'], width=3),
        marker=dict(size=8, color=COLORS['warning']),
        hovertemplate='<b>%{x}</b><br>Rate: %{y:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        barmode='overlay',
        title=dict(text='Loan Acceptance by Income Band', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Income Band', gridcolor='#30363D'),
        yaxis=dict(title='Customer Count', gridcolor='#30363D'),
        yaxis2=dict(title='Acceptance Rate (%)', overlaying='y', side='right',
                    gridcolor='#30363D', range=[0, 100]),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
        height=420,
    )
    return fig
