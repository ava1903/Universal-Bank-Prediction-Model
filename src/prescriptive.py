import anthropic
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    'bg': '#0D1117', 'card': '#161B22', 'text': '#E6EDF3',
    'loan_yes': '#00C49F', 'loan_no': '#FF6B6B',
    'accent': '#2196F3', 'warning': '#FF9800', 'purple': '#9C27B0',
    'success': '#4CAF50',
}

EDUCATION_MAP = {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'}
ONLINE_MAP = {0: 'does not use online banking', 1: 'uses online banking'}
CC_MAP = {0: 'does not have a UniversalBank credit card', 1: 'has a UniversalBank credit card'}
SEC_MAP = {0: 'does not hold a securities account', 1: 'holds a securities account'}
CD_MAP = {0: 'does not have a CD account', 1: 'has a CD account with UniversalBank'}

def build_customer_context(customer: dict, prob: float) -> str:
    edu = EDUCATION_MAP.get(customer.get('Education', 1), 'Unknown')
    income = customer.get('Income', 0)
    age = customer.get('Age', 0)
    family = customer.get('Family', 1)
    ccavg = customer.get('CCAvg', 0)
    mortgage = customer.get('Mortgage', 0)
    online = ONLINE_MAP.get(customer.get('Online', 0))
    cc = CC_MAP.get(customer.get('CreditCard', 0))
    sec = SEC_MAP.get(customer.get('Securities Account', 0))
    cd = CD_MAP.get(customer.get('CD Account', 0))
    family_str = f"family of {family}" if family > 1 else "single individual"

    context = f"""
Customer Profile:
- Age: {age} years old
- Education: {edu} degree holder
- Annual Income: ${income},000
- Family: {family_str}
- Monthly CC Spending: ${ccavg}K
- Mortgage: {'$' + str(mortgage) + 'K outstanding' if mortgage > 0 else 'No mortgage'}
- Banking: {online}, {cc}
- Investments: {sec}, {cd}
- AI Loan Interest Score: {prob*100:.1f}% probability
"""
    return context.strip()

def generate_personalized_message(customer: dict, prob: float, api_key: str = None) -> str:
    context = build_customer_context(customer, prob)
    prompt = f"""You are a senior relationship manager at UniversalBank. 
Based on this customer profile, write a warm, personalized, and compelling outreach message 
to offer them a personal loan. The message should:
1. Address them personally based on their life stage and financial profile
2. Highlight specific benefits relevant to their situation
3. Mention relevant loan features (competitive rates, flexible tenure, quick approval)
4. Reference their existing relationship with the bank if applicable
5. End with a clear, non-pushy call to action
6. Be 150-200 words, professional yet warm

{context}

Write only the message text, no subject line or formatting labels."""

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return _fallback_message(customer, prob)

def _fallback_message(customer: dict, prob: float) -> str:
    edu = EDUCATION_MAP.get(customer.get('Education', 1), 'valued')
    income = customer.get('Income', 0)
    family = customer.get('Family', 1)
    has_cd = customer.get('CD Account', 0)
    has_cc = customer.get('CreditCard', 0)

    relationship = ""
    if has_cd:
        relationship = "As a valued CD account holder, "
    elif has_cc:
        relationship = "As a loyal UniversalBank credit card member, "

    family_note = ""
    if family >= 3:
        family_note = f"We understand that managing finances for a family of {family} comes with unique responsibilities. "

    return f"""Dear Valued Customer,

{relationship}we at UniversalBank have a special personal loan offer tailored just for you.

{family_note}Given your {edu.lower()} background and strong financial profile with an annual income of ${income}K, you qualify for our premium personal loan with:

✓ Competitive interest rates starting at 8.5% p.a.
✓ Loan amount up to ${min(income * 3, 500)}K
✓ Flexible repayment tenure of 12-60 months
✓ Zero pre-payment charges
✓ Approval within 24 hours

Whether it's home renovation, education, travel, or any personal milestone — we're here to make it happen.

To explore your personalized offer, simply call us at 1-800-UNIBANK or visit your nearest branch.

Warm regards,
UniversalBank Relationship Team"""

def optimal_threshold_chart(results: dict, model_name: str):
    res = results[model_name]
    y_test, y_prob = res['y_test'], res['y_prob']

    thresholds = np.arange(0.1, 0.9, 0.02)
    precisions, recalls, f1s, accuracies = [], [], [], []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        accuracies.append(accuracy_score(y_test, preds))

    fig = go.Figure()
    for name, vals, color in [
        ('Precision', precisions, COLORS['accent']),
        ('Recall', recalls, COLORS['loan_yes']),
        ('F1 Score', f1s, COLORS['warning']),
        ('Accuracy', accuracies, COLORS['purple']),
    ]:
        fig.add_trace(go.Scatter(
            x=thresholds, y=vals, name=name, mode='lines',
            line=dict(width=2, color=color),
            hovertemplate=f'<b>{name}</b><br>Threshold: %{{x:.2f}}<br>Score: %{{y:.3f}}<extra></extra>',
        ))
    best_t = thresholds[np.argmax(f1s)]
    fig.add_vline(x=best_t, line_dash='dash', line_color='white',
                  annotation_text=f'Best F1 @ {best_t:.2f}',
                  annotation_font_color='white')
    fig.update_layout(
        title=dict(text=f'Decision Threshold Optimization — {model_name}', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Classification Threshold', gridcolor='#30363D'),
        yaxis=dict(title='Score', gridcolor='#30363D', range=[0, 1.05]),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=420,
    )
    return fig

def high_value_segments_chart(df: pd.DataFrame):
    seg = df.groupby(['Income_Band', 'Education_Label'], observed=True).agg(
        Loan_Rate=('Personal Loan', 'mean'),
        Count=('Personal Loan', 'count'),
    ).reset_index()
    seg['Loan_Rate_Pct'] = (seg['Loan_Rate'] * 100).round(1)
    seg['Expected_Converts'] = (seg['Count'] * seg['Loan_Rate']).round(0).astype(int)

    fig = px.treemap(
        seg, path=['Education_Label', 'Income_Band'],
        values='Count', color='Loan_Rate_Pct',
        color_continuous_scale='YlOrRd',
        hover_data={'Loan_Rate_Pct': ':.1f', 'Expected_Converts': True},
        title='High-Value Target Segments (Education × Income)',
        custom_data=['Loan_Rate_Pct', 'Expected_Converts'],
    )
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Customers: %{value:,}<br>Loan Rate: %{customdata[0]:.1f}%<br>Expected Converts: %{customdata[1]}<extra></extra>',
        textinfo='label+value',
    )
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        title=dict(font=dict(size=16, color=COLORS['text'])),
        coloraxis_colorbar=dict(title='Loan Rate %', tickfont=dict(color=COLORS['text'])),
        height=480,
    )
    return fig

def campaign_roi_simulator(df: pd.DataFrame, results: dict, best_model: str):
    y_prob = results[best_model]['y_prob']
    thresholds = np.arange(0.1, 0.95, 0.05)
    total = len(y_prob)

    rows = []
    for t in thresholds:
        targeted = (y_prob >= t).sum()
        y_true = results[best_model]['y_test']
        tp = ((y_prob >= t) & (y_true == 1)).sum()
        rows.append({
            'Threshold': round(t, 2),
            'Targeted': int(targeted),
            'Target_Pct': round(targeted / total * 100, 1),
            'True_Positives': int(tp),
            'Precision': round(tp / targeted * 100, 1) if targeted > 0 else 0,
        })
    tbl = pd.DataFrame(rows)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=tbl['Threshold'], y=tbl['Targeted'],
        name='Customers Targeted', marker_color=COLORS['accent'], opacity=0.7,
        hovertemplate='Threshold: %{x}<br>Targeted: %{y:,}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tbl['Threshold'], y=tbl['Precision'],
        name='Precision of Targeting (%)', mode='lines+markers',
        line=dict(color=COLORS['loan_yes'], width=3),
        marker=dict(size=7),
        hovertemplate='Threshold: %{x}<br>Precision: %{y:.1f}%<extra></extra>',
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=tbl['Threshold'], y=tbl['True_Positives'],
        name='True Loan Takers Captured', mode='lines+markers',
        line=dict(color=COLORS['warning'], width=3, dash='dot'),
        marker=dict(size=7),
        hovertemplate='Threshold: %{x}<br>True Positives: %{y}<extra></extra>',
    ), secondary_y=False)
    fig.update_layout(
        title=dict(text='Campaign ROI Simulator — Threshold vs Targeting Efficiency', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Decision Threshold', gridcolor='#30363D'),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.12),
        height=440,
    )
    fig.update_yaxes(title_text='Number of Customers', gridcolor='#30363D', secondary_y=False)
    fig.update_yaxes(title_text='Precision (%)', gridcolor='#30363D', secondary_y=True)
    return fig
