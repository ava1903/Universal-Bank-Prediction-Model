import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    'primary': '#1E3A5F', 'accent': '#2196F3', 'success': '#4CAF50',
    'danger': '#F44336', 'warning': '#FF9800', 'purple': '#9C27B0',
    'teal': '#009688', 'bg': '#0D1117', 'card': '#161B22',
    'text': '#E6EDF3', 'loan_yes': '#00C49F', 'loan_no': '#FF6B6B',
}

def loan_by_education_family(df):
    pivot = df.groupby(['Education_Label', 'Family'])['Personal Loan'].mean().unstack(fill_value=0) * 100
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'Family {c}' for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale='YlOrRd',
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont=dict(size=12),
        hovertemplate='<b>%{y}</b> | <b>%{x}</b><br>Loan Rate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='Loan Rate %', tickfont=dict(color=COLORS['text'])),
    ))
    fig.update_layout(
        title=dict(text='Loan Acceptance Rate: Education × Family Size', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Family Size'),
        yaxis=dict(title='Education Level'),
        height=380,
    )
    return fig

def income_ccavg_loan_density(df):
    loan_yes = df[df['Personal Loan'] == 1]
    loan_no = df[df['Personal Loan'] == 0].sample(n=min(500, len(df[df['Personal Loan']==0])), random_state=42)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loan_no['Income'], y=loan_no['CCAvg'],
        mode='markers',
        name='No Loan',
        marker=dict(color=COLORS['loan_no'], size=6, opacity=0.4),
        hovertemplate='Income: %{x}K<br>CC Avg: $%{y}K<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=loan_yes['Income'], y=loan_yes['CCAvg'],
        mode='markers',
        name='Accepted Loan',
        marker=dict(color=COLORS['loan_yes'], size=8, opacity=0.8,
                    symbol='diamond'),
        hovertemplate='Income: %{x}K<br>CC Avg: $%{y}K<extra></extra>',
    ))
    fig.add_hline(y=df[df['Personal Loan']==1]['CCAvg'].median(),
                  line_dash='dash', line_color=COLORS['loan_yes'], opacity=0.6,
                  annotation_text='Loan Accepted Median CCAvg',
                  annotation_font_color=COLORS['loan_yes'])
    fig.add_vline(x=df[df['Personal Loan']==1]['Income'].median(),
                  line_dash='dash', line_color=COLORS['loan_yes'], opacity=0.6,
                  annotation_text='Loan Accepted Median Income',
                  annotation_font_color=COLORS['loan_yes'])
    fig.update_layout(
        title=dict(text='Income vs CC Spending — Loan Acceptance Zones', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Annual Income ($000)', gridcolor='#30363D'),
        yaxis=dict(title='Avg CC Spending ($000/month)', gridcolor='#30363D'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=420,
    )
    return fig

def mortgage_vs_loan(df):
    df2 = df.copy()
    df2['Has Mortgage'] = (df2['Mortgage'] > 0).map({True: 'Has Mortgage', False: 'No Mortgage'})
    df2['Mortgage_Band'] = pd.cut(df2['Mortgage'], bins=[-1, 0, 100, 300, 500, 1000],
                                   labels=['None', '<100K', '100-300K', '300-500K', '500K+'])
    tbl = df2.groupby('Mortgage_Band', observed=True)['Personal Loan'].agg(['mean', 'count']).reset_index()
    tbl.columns = ['Mortgage Band', 'Loan Rate', 'Count']
    tbl['Loan Rate %'] = (tbl['Loan Rate'] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tbl['Mortgage Band'].astype(str),
        y=tbl['Loan Rate %'],
        marker=dict(
            color=tbl['Loan Rate %'],
            colorscale='Teal',
            showscale=True,
            colorbar=dict(title='Loan %', tickfont=dict(color=COLORS['text'])),
        ),
        text=tbl['Loan Rate %'].apply(lambda x: f'{x}%'),
        textposition='outside',
        textfont=dict(color=COLORS['text']),
        hovertemplate='<b>%{x}</b><br>Loan Rate: %{y:.1f}%<br>Count: ' +
                      tbl['Count'].astype(str) + '<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Loan Acceptance Rate by Mortgage Level', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Mortgage Band', gridcolor='#30363D'),
        yaxis=dict(title='Loan Acceptance Rate (%)', gridcolor='#30363D', range=[0, 50]),
        height=400,
    )
    return fig

def cd_securities_impact(df):
    groups = df.groupby(['CD Account', 'Securities Account'])['Personal Loan'].agg(['mean', 'count']).reset_index()
    groups.columns = ['CD Account', 'Securities Account', 'Loan Rate', 'Count']
    groups['CD Label'] = groups['CD Account'].map({0: 'No CD', 1: 'Has CD'})
    groups['Sec Label'] = groups['Securities Account'].map({0: 'No Securities', 1: 'Has Securities'})
    groups['Label'] = groups['CD Label'] + ' + ' + groups['Sec Label']
    groups['Loan Rate %'] = (groups['Loan Rate'] * 100).round(1)

    fig = go.Figure(go.Bar(
        x=groups['Label'],
        y=groups['Loan Rate %'],
        marker=dict(
            color=[COLORS['loan_no'], COLORS['warning'], COLORS['accent'], COLORS['loan_yes']],
            line=dict(color='#0D1117', width=2),
        ),
        text=groups['Loan Rate %'].apply(lambda x: f'{x}%'),
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=13),
        hovertemplate='<b>%{x}</b><br>Loan Rate: %{y:.1f}%<br>N=%{customdata}<extra></extra>',
        customdata=groups['Count'],
    ))
    fig.update_layout(
        title=dict(text='Loan Rate by CD & Securities Account Combinations', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Account Combination', gridcolor='#30363D'),
        yaxis=dict(title='Loan Acceptance Rate (%)', gridcolor='#30363D', range=[0, 60]),
        height=420,
    )
    return fig

def age_group_loan_funnel(df):
    tbl = df.groupby('Age_Group', observed=True)['Personal Loan'].agg(['sum', 'count']).reset_index()
    tbl.columns = ['Age Group', 'Accepted', 'Total']
    tbl['Rejected'] = tbl['Total'] - tbl['Accepted']
    tbl['Rate'] = (tbl['Accepted'] / tbl['Total'] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Rejected', x=tbl['Age Group'].astype(str), y=tbl['Rejected'],
        marker_color=COLORS['loan_no'], opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Rejected: %{y:,}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Accepted', x=tbl['Age Group'].astype(str), y=tbl['Accepted'],
        marker_color=COLORS['loan_yes'], opacity=0.9,
        hovertemplate='<b>%{x}</b><br>Accepted: %{y:,}<extra></extra>',
        text=tbl['Rate'].apply(lambda x: f'{x}%'),
        textposition='inside', textfont=dict(color='white', size=12),
    ))
    fig.update_layout(
        barmode='stack',
        title=dict(text='Loan Acceptance by Age Group', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Age Group', gridcolor='#30363D'),
        yaxis=dict(title='Number of Customers', gridcolor='#30363D'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=400,
    )
    return fig

def parallel_coordinates_chart(df):
    sample = df.sample(n=min(1000, len(df)), random_state=42).copy()
    sample['Loan Color'] = sample['Personal Loan'].map({0: 0, 1: 1})
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=sample['Loan Color'],
            colorscale=[[0, COLORS['loan_no']], [1, COLORS['loan_yes']]],
            showscale=True,
            colorbar=dict(title='Loan', tickvals=[0, 1], ticktext=['No', 'Yes'],
                          tickfont=dict(color=COLORS['text'])),
        ),
        dimensions=[
            dict(label='Age', values=sample['Age'], range=[20, 70]),
            dict(label='Income', values=sample['Income'], range=[0, 250]),
            dict(label='CCAvg', values=sample['CCAvg'], range=[0, 15]),
            dict(label='Education', values=sample['Education'], range=[1, 3],
                 tickvals=[1, 2, 3], ticktext=['UG', 'Grad', 'Adv']),
            dict(label='Family', values=sample['Family'], range=[1, 4]),
            dict(label='Mortgage', values=sample['Mortgage'], range=[0, 500]),
        ],
    ))
    fig.update_layout(
        title=dict(text='Parallel Coordinates — Customer Profile by Loan Status', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=450,
    )
    return fig
