import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve,
                              accuracy_score, f1_score)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

COLORS = {
    'bg': '#0D1117', 'card': '#161B22', 'text': '#E6EDF3',
    'loan_yes': '#00C49F', 'loan_no': '#FF6B6B',
    'accent': '#2196F3', 'warning': '#FF9800', 'purple': '#9C27B0',
    'success': '#4CAF50',
}
MODEL_COLORS = {
    'Logistic Regression': '#2196F3',
    'Decision Tree': '#FF9800',
    'Random Forest': '#00C49F',
    'XGBoost': '#9C27B0',
}

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                                 random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           scale_pos_weight=9, eval_metric='logloss',
                                           random_state=42, verbosity=0)
    except ImportError:
        pass
    return models

def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}
    models = get_models()

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model, 'scaler': scaler if name == 'Logistic Regression' else None,
            'accuracy': acc, 'f1': f1, 'roc_auc': roc_auc,
            'fpr': fpr, 'tpr': tpr, 'cm': cm,
            'report': report, 'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
            'feature_names': X.columns.tolist(),
        }

    # save best model
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best = results[best_name]
    joblib.dump({'model': best['model'], 'scaler': best['scaler'],
                 'features': best['feature_names'], 'name': best_name},
                os.path.join(MODELS_DIR, 'best_model.pkl'))

    return results, best_name, X_test, y_test

def model_comparison_chart(results):
    names = list(results.keys())
    metrics = {
        'Accuracy': [results[n]['accuracy'] for n in names],
        'F1 Score': [results[n]['f1'] for n in names],
        'ROC AUC': [results[n]['roc_auc'] for n in names],
    }
    metric_colors = [COLORS['accent'], COLORS['loan_yes'], COLORS['warning']]
    fig = go.Figure()
    for i, (metric, vals) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            name=metric, x=names,
            y=[round(v * 100, 2) for v in vals],
            marker_color=metric_colors[i],
            text=[f'{v*100:.1f}%' for v in vals],
            textposition='outside', textfont=dict(color=COLORS['text']),
            hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:.2f}%<extra></extra>',
        ))
    fig.update_layout(
        barmode='group',
        title=dict(text='Model Performance Comparison', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(gridcolor='#30363D'),
        yaxis=dict(title='Score (%)', gridcolor='#30363D', range=[0, 110]),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
        height=420,
    )
    return fig

def roc_curves_chart(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Random Classifier', showlegend=True,
    ))
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            x=res['fpr'], y=res['tpr'], mode='lines',
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            line=dict(width=2.5, color=MODEL_COLORS.get(name, COLORS['accent'])),
            hovertemplate=f'<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='ROC Curves — All Models', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='False Positive Rate', gridcolor='#30363D'),
        yaxis=dict(title='True Positive Rate', gridcolor='#30363D'),
        legend=dict(bgcolor='rgba(0,0,0,0)', x=0.6, y=0.1),
        height=430,
    )
    return fig

def confusion_matrix_chart(results, selected_model):
    cm = results[selected_model]['cm']
    labels = ['No Loan (0)', 'Loan (1)']
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=j, y=i, text=str(cm[i, j]),
                font=dict(size=20, color='white'), showarrow=False,
            ))
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, '#161B22'], [0.5, '#1E4D8C'], [1, COLORS['loan_yes']]],
        showscale=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    ))
    fig.update_layout(
        annotations=annotations,
        title=dict(text=f'Confusion Matrix — {selected_model}', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Predicted'),
        yaxis=dict(title='Actual'),
        height=380,
    )
    return fig

def feature_importance_chart(results, selected_model):
    res = results[selected_model]
    model = res['model']
    feat_names = res['feature_names']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return go.Figure()

    idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feat_names[i] for i in idx],
        orientation='h',
        marker=dict(
            color=importances[idx],
            colorscale='Teal',
            showscale=True,
            colorbar=dict(title='Importance', tickfont=dict(color=COLORS['text'])),
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=f'Feature Importance — {selected_model}', font=dict(size=16, color=COLORS['text'])),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title='Importance Score', gridcolor='#30363D'),
        yaxis=dict(gridcolor='#30363D'),
        height=420,
    )
    return fig

def prediction_probability_gauge(prob):
    color = COLORS['loan_yes'] if prob >= 0.5 else COLORS['loan_no']
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=round(prob * 100, 1),
        title=dict(text='Loan Interest Probability', font=dict(size=16, color=COLORS['text'])),
        number=dict(suffix='%', font=dict(size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS['text'],
                      tickfont=dict(color=COLORS['text'])),
            bar=dict(color=color, thickness=0.3),
            bgcolor='#161B22',
            bordercolor='#30363D',
            steps=[
                dict(range=[0, 30], color='#1a1a2e'),
                dict(range=[30, 60], color='#16213e'),
                dict(range=[60, 100], color='#0f3460'),
            ],
            threshold=dict(line=dict(color='white', width=3), thickness=0.8, value=50),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=300,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig

def predict_customer(customer_data: dict, results: dict, best_model_name: str):
    res = results[best_model_name]
    model = res['model']
    scaler = res['scaler']
    feat_names = res['feature_names']

    df_input = pd.DataFrame([customer_data])[feat_names]

    if scaler:
        df_scaled = scaler.transform(df_input)
        prob = model.predict_proba(df_scaled)[0][1]
        pred = model.predict(df_scaled)[0]
    else:
        prob = model.predict_proba(df_input)[0][1]
        pred = model.predict(df_input)[0]

    return int(pred), float(prob)
