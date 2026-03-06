# 🏦 UniversalBank Loan Intelligence Dashboard

A full-stack AI-powered Streamlit dashboard to predict customer interest in personal loans and generate personalized outreach messages using Claude AI.

## 🎯 Objective

Predict whether a bank customer will accept a personal loan offer, then generate a hyper-personalized AI message for likely acceptors — enabling targeted, data-driven marketing campaigns.

---

## 📊 Four Layers of Analytics

| Layer | Description |
|-------|-------------|
| **📊 Descriptive** | Who are our customers? Distributions, demographics, spending patterns |
| **🔍 Diagnostic** | Why do they accept loans? Feature analysis, segment heatmaps, behavioral patterns |
| **🤖 Predictive** | ML models (LR, DT, RF, XGBoost) with ROC curves, confusion matrices, feature importance |
| **💬 Prescriptive** | Target segments, campaign ROI simulator, AI-generated personalized messages |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/universalbank-loan-dashboard.git
cd universalbank-loan-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Dataset
Place `UniversalBank.csv` in the `data/` folder.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Add your `ANTHROPIC_API_KEY` in **Secrets** (Settings → Secrets):
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## 🤖 AI-Powered Personalized Messages

The dashboard uses the **Claude API (claude-sonnet-4-20250514)** to generate personalized loan outreach messages based on each customer's:
- Age, income, and family profile
- Education level
- Existing bank relationships (CD, Securities, Credit Card)
- Predicted loan interest probability

To use AI messages, add your Anthropic API key in the sidebar (or set `ANTHROPIC_API_KEY` as a Streamlit secret for deployment).

---

## 📁 Project Structure

```
universalbank-loan-dashboard/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md
├── data/
│   └── UniversalBank.csv     ← Dataset (5,000 customers)
├── models/
│   └── best_model.pkl        ← Auto-saved best model
└── src/
    ├── data_loader.py        ← Data loading & feature engineering
    ├── descriptive.py        ← Descriptive analysis charts
    ├── diagnostic.py         ← Diagnostic analysis charts
    ├── predictive.py         ← ML model training & evaluation
    └── prescriptive.py       ← Prescriptive analysis & AI messaging
```

---

## 🛠 Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble of 200 trees |
| XGBoost | Gradient boosted trees (best performance) |

All models handle **class imbalance** (only 9.6% loan acceptors) via `class_weight='balanced'` or `scale_pos_weight`.

---

## 📈 Key Features

- **Interactive Plotly charts** with drill-down capability
- **Parallel coordinates** for multi-variable exploration
- **ROC curves** comparing all 4 models
- **Decision threshold optimizer** for campaign precision/recall tradeoff
- **Campaign ROI simulator** — optimize targeting threshold
- **Treemap** of high-value customer segments
- **Real-time customer prediction** with probability gauge
- **Claude AI message generation** for predicted loan takers

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
plotly>=5.18.0
joblib>=1.3.0
anthropic>=0.21.0
```

---

## 📝 Dataset

**UniversalBank.csv** — 5,000 customers of Universal Bank

| Feature | Description |
|---------|-------------|
| Age | Customer age |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| Family | Family size |
| CCAvg | Avg monthly credit card spending ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Home mortgage value ($000) |
| Personal Loan | **Target** — Did customer accept loan? |
| Securities Account | Has securities account? |
| CD Account | Has certificate of deposit? |
| Online | Uses internet banking? |
| CreditCard | Has UniversalBank credit card? |

> **Note:** `ID` and `ZIP Code` are excluded from modeling.

---

## 👤 Author

Built with ❤️ using Streamlit, Plotly, Scikit-learn, XGBoost, and Claude AI.
