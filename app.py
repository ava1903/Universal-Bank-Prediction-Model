import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, get_features_target
from src.descriptive import (loan_distribution_donut, age_income_scatter, correlation_heatmap,
                               feature_distributions, categorical_breakdown, income_band_loan_rate)
from src.diagnostic import (loan_by_education_family, income_ccavg_loan_density,
                              mortgage_vs_loan, cd_securities_impact,
                              age_group_loan_funnel, parallel_coordinates_chart)
from src.predictive import (train_all_models, model_comparison_chart, roc_curves_chart,
                              confusion_matrix_chart, feature_importance_chart,
                              prediction_probability_gauge, predict_customer)
from src.prescriptive import (generate_personalized_message, optimal_threshold_chart,
                                high_value_segments_chart, campaign_roi_simulator)
from src.feature_importance import compute_all_importances
from src.feature_importance_charts import (method_comparison_grouped, composite_score_bar,
                                            radar_chart, pearson_vs_truth_divergence,
                                            drop_one_waterfall, perm_importance_with_error,
                                            verdict_summary_table)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniversalBank | Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    [data-testid="stSidebar"] .stRadio label { color: #E6EDF3 !important; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #161B22, #1C2128);
        border: 1px solid #30363D; border-radius: 12px;
        padding: 16px; margin: 4px 0;
    }
    [data-testid="metric-container"] label { color: #8B949E !important; font-size: 13px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-size: 28px !important; font-weight: 700 !important; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 13px !important; }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1E3A5F, #0D1117);
        border-left: 4px solid #2196F3;
        padding: 12px 20px; border-radius: 0 8px 8px 0;
        margin: 20px 0 16px 0;
    }
    .section-header h3 { color: #E6EDF3; margin: 0; font-size: 18px; font-weight: 600; }
    .section-header p { color: #8B949E; margin: 4px 0 0 0; font-size: 13px; }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #161B22, #1C2128);
        border: 1px solid #30363D; border-radius: 12px;
        padding: 16px; margin: 8px 0;
    }
    .insight-card.positive { border-left: 4px solid #00C49F; }
    .insight-card.warning  { border-left: 4px solid #FF9800; }
    .insight-card.info     { border-left: 4px solid #2196F3; }
    .insight-card h4 { color: #E6EDF3; margin: 0 0 6px 0; font-size: 14px; font-weight: 600; }
    .insight-card p  { color: #8B949E; margin: 0; font-size: 13px; line-height: 1.5; }
    
    /* Prediction result */
    .pred-yes {
        background: linear-gradient(135deg, #0D2818, #0F3D20);
        border: 2px solid #00C49F; border-radius: 16px;
        padding: 24px; text-align: center;
    }
    .pred-no {
        background: linear-gradient(135deg, #2D0E0E, #3D1212);
        border: 2px solid #FF6B6B; border-radius: 16px;
        padding: 24px; text-align: center;
    }
    
    /* Message box */
    .message-box {
        background: linear-gradient(135deg, #161B22, #1C2128);
        border: 1px solid #2196F3; border-radius: 12px;
        padding: 24px; margin-top: 16px;
        white-space: pre-wrap; line-height: 1.7;
        color: #E6EDF3; font-size: 14px;
    }
    
    /* Nav pills */
    .nav-pill {
        display: inline-block; padding: 8px 16px;
        background: #21262D; border: 1px solid #30363D;
        border-radius: 20px; margin: 4px; color: #8B949E; font-size: 13px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #161B22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8B949E !important; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background-color: #1E3A5F !important; color: #E6EDF3 !important; }
    
    /* Selectbox / Inputs */
    .stSelectbox > div > div { background-color: #21262D; border-color: #30363D; color: #E6EDF3; }
    .stSlider > div { color: #E6EDF3; }
    .stNumberInput input { background-color: #21262D; border-color: #30363D; color: #E6EDF3; }
    .stTextInput input { background-color: #21262D; border-color: #30363D; color: #E6EDF3; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F, #2196F3);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(33,150,243,0.4); }
    
    /* Divider */
    hr { border-color: #30363D; margin: 24px 0; }
    
    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* DataFrames */
    .stDataFrame { border: 1px solid #30363D; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
@st.cache_data
def load_cached_data():
    return load_data()

@st.cache_resource
def compute_cached_importances():
    return compute_all_importances()
    df = load_cached_data()
    X, y = get_features_target(df)
    return train_all_models(X, y)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:40px;">🏦</div>
        <div style="font-size:18px; font-weight:700; color:#E6EDF3;">UniversalBank</div>
        <div style="font-size:12px; color:#8B949E; margin-top:4px;">Loan Intelligence Dashboard</div>
    </div>
    <hr style="border-color:#30363D; margin:12px 0 20px 0;">
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Overview",
        "📊  Descriptive Analysis",
        "🔍  Diagnostic Analysis",
        "🧬  Feature Importance",
        "🤖  Predictive Modeling",
        "💬  Prescriptive & Messaging",
    ], label_visibility='collapsed')

    st.markdown("<hr style='border-color:#30363D;'>", unsafe_allow_html=True)

    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-...",
                             help="Required for AI-generated personalized messages")

    st.markdown("""
    <div style="margin-top:20px; padding:12px; background:#21262D; border-radius:8px;">
        <div style="color:#8B949E; font-size:11px; line-height:1.6;">
        <b style="color:#E6EDF3;">Dataset:</b> UniversalBank.csv<br>
        <b style="color:#E6EDF3;">Records:</b> 5,000 customers<br>
        <b style="color:#E6EDF3;">Target:</b> Personal Loan<br>
        <b style="color:#E6EDF3;">Models:</b> LR · DT · RF · XGB
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Load Data & Models ────────────────────────────────────────────────────────
df = load_cached_data()
X, y = get_features_target(df)
feature_hash = str(X.columns.tolist())

with st.spinner("Training models on first load..."):
    results, best_model_name, X_test, y_test = train_cached_models(feature_hash)

# ─── PAGE: OVERVIEW ───────────────────────────────────────────────────────────
if page == "🏠  Overview":
    st.markdown("""
    <div style="padding: 32px 0 24px 0;">
        <h1 style="color:#E6EDF3; font-size:32px; font-weight:700; margin:0;">
            Loan Customer Intelligence
        </h1>
        <p style="color:#8B949E; font-size:16px; margin-top:8px;">
            Predict customer interest in personal loans and generate personalized outreach
        </p>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    accepted = df['Personal Loan'].sum()
    avg_income = df['Income'].mean()
    avg_ccavg = df['CCAvg'].mean()
    best_auc = results[best_model_name]['roc_auc']

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Loan Acceptances", f"{accepted:,}", f"{accepted/total*100:.1f}% rate")
    c3.metric("Avg. Annual Income", f"${avg_income:.0f}K")
    c4.metric("Avg. CC Spending", f"${avg_ccavg:.2f}K/mo")
    c5.metric("Best Model AUC", f"{best_auc:.3f}", f"{best_model_name}")

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.plotly_chart(loan_distribution_donut(df), use_container_width=True)
    with col2:
        st.plotly_chart(income_band_loan_rate(df), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <h3>🎯 Key Insights at a Glance</h3>
        <p>Data-driven findings from 5,000 customers</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        income_threshold = df[df['Personal Loan']==1]['Income'].quantile(0.25)
        st.markdown(f"""
        <div class="insight-card positive">
            <h4>💰 Income is the #1 Driver</h4>
            <p>Customers with income above ${income_threshold:.0f}K show dramatically higher loan acceptance. 
            Loan takers earn {df[df['Personal Loan']==1]['Income'].mean():.0f}K vs 
            {df[df['Personal Loan']==0]['Income'].mean():.0f}K on average.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        cd_rate = df[df['CD Account']==1]['Personal Loan'].mean()*100
        no_cd_rate = df[df['CD Account']==0]['Personal Loan'].mean()*100
        st.markdown(f"""
        <div class="insight-card warning">
            <h4>🏦 CD Account = Strong Signal</h4>
            <p>CD account holders accept loans at {cd_rate:.0f}% vs only {no_cd_rate:.0f}% for non-holders — 
            a {cd_rate/no_cd_rate:.1f}x higher conversion rate. Existing relationship matters.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        adv_rate = df[df['Education']==3]['Personal Loan'].mean()*100
        ug_rate = df[df['Education']==1]['Personal Loan'].mean()*100
        st.markdown(f"""
        <div class="insight-card info">
            <h4>🎓 Education Amplifies Intent</h4>
            <p>Advanced/Professional degree holders accept loans at {adv_rate:.0f}% vs 
            {ug_rate:.0f}% for undergrads. Higher education correlates with both income 
            and willingness to leverage credit.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #161B22, #1C2128); border: 1px solid #30363D; 
                border-radius: 12px; padding: 20px; text-align:center;">
        <div style="color:#8B949E; font-size:13px; margin-bottom:12px;">Navigate the Dashboard</div>
        <span class="nav-pill">📊 Descriptive — Who are our customers?</span>
        <span class="nav-pill">🔍 Diagnostic — Why do they take loans?</span>
        <span class="nav-pill">🧬 Feature Importance — What actually predicts?</span>
        <span class="nav-pill">🤖 Predictive — Who will take a loan?</span>
        <span class="nav-pill">💬 Prescriptive — How do we engage them?</span>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: DESCRIPTIVE ────────────────────────────────────────────────────────
elif page == "📊  Descriptive Analysis":
    st.markdown("""
    <div class="section-header">
        <h3>📊 Descriptive Analysis</h3>
        <p>Understanding who our customers are — distributions, profiles, and patterns</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔥 Correlations", "🍩 Categorical", "💰 Income Analysis"])

    with tab1:
        st.plotly_chart(feature_distributions(df), use_container_width=True)
        st.plotly_chart(age_income_scatter(df), use_container_width=True)

    with tab2:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        num_df = df.select_dtypes(include='number')
        corr_with_target = num_df.corrwith(df['Personal Loan']).drop('Personal Loan').abs().sort_values(ascending=False)
        with c1:
            st.markdown(f"""
            <div class="insight-card positive"><h4>🏆 Top Predictor</h4>
            <p><b>{corr_with_target.index[0]}</b> has the highest correlation with loan acceptance 
            (r = {corr_with_target.iloc[0]:.3f})</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="insight-card info"><h4>2nd Predictor</h4>
            <p><b>{corr_with_target.index[1]}</b> is the second most important feature 
            (r = {corr_with_target.iloc[1]:.3f})</p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="insight-card warning"><h4>3rd Predictor</h4>
            <p><b>{corr_with_target.index[2]}</b> ranks 3rd by correlation 
            (r = {corr_with_target.iloc[2]:.3f})</p></div>""", unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(categorical_breakdown(df), use_container_width=True)
        st.markdown("##### 📋 Summary Statistics")
        summary = df[['Age', 'Income', 'CCAvg', 'Mortgage', 'Experience', 'Family']].describe().round(2)
        st.dataframe(summary, use_container_width=True)

    with tab4:
        st.plotly_chart(income_band_loan_rate(df), use_container_width=True)
        tbl = df.groupby('Income_Band', observed=True)['Personal Loan'].agg(
            Total='count', Accepted='sum').reset_index()
        tbl['Rejection'] = tbl['Total'] - tbl['Accepted']
        tbl['Rate %'] = (tbl['Accepted'] / tbl['Total'] * 100).round(1)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ─── PAGE: DIAGNOSTIC ─────────────────────────────────────────────────────────
elif page == "🔍  Diagnostic Analysis":
    st.markdown("""
    <div class="section-header">
        <h3>🔍 Diagnostic Analysis</h3>
        <p>Understanding WHY customers accept loans — drivers, segments, and root causes</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Segment Analysis", "📈 Behavioral Patterns", "🔗 Multi-Variable"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(loan_by_education_family(df), use_container_width=True)
        with col2:
            st.plotly_chart(age_group_loan_funnel(df), use_container_width=True)
        st.plotly_chart(cd_securities_impact(df), use_container_width=True)

    with tab2:
        st.plotly_chart(income_ccavg_loan_density(df), use_container_width=True)
        st.plotly_chart(mortgage_vs_loan(df), use_container_width=True)

    with tab3:
        st.plotly_chart(parallel_coordinates_chart(df), use_container_width=True)
        st.markdown("""
        <div class="insight-card info">
            <h4>💡 How to read this chart</h4>
            <p>Each line represents a customer. <span style="color:#00C49F;">Green lines</span> are loan acceptors, 
            <span style="color:#FF6B6B;">red lines</span> are non-acceptors. 
            Drag on any axis to filter by that range and identify high-conversion segments. 
            Notice how high-income, high-CCAvg, advanced-education customers cluster at the top.</p>
        </div>""", unsafe_allow_html=True)

# ─── PAGE: FEATURE IMPORTANCE ─────────────────────────────────────────────────
elif page == "🧬  Feature Importance":
    st.markdown("""
    <div class="section-header">
        <h3>🧬 Feature Importance — Beyond Pearson Correlation</h3>
        <p>6-method triangulated analysis: Pearson, Mutual Info, Random Forest, Permutation, Drop-One AUC, LR Coefficients</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running 6-method importance analysis (takes ~15s first time)..."):
        result_df, raw_df, baseline_auc = compute_cached_importances()

    # Why correlation alone isn't enough
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a0a0a,#2d1010);border:1px solid #FF4444;
                border-radius:12px;padding:20px;margin-bottom:20px;">
        <h4 style="color:#FF6B6B;margin:0 0 10px 0;">⚠️ Why Pearson Correlation Alone is Unreliable Here</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;color:#E6EDF3;font-size:13px;">
            <div><b style="color:#FF9800;">1. Binary target problem</b><br>
            Personal Loan is 0/1. Pearson assumes continuous normal distributions — 
            it systematically underestimates binary predictors like CD Account.</div>
            <div><b style="color:#FF9800;">2. Misses non-linearity</b><br>
            Income's effect isn't linear — it has a threshold (~$100K) where loan acceptance 
            jumps sharply. Pearson can't capture this inflection.</div>
            <div><b style="color:#FF9800;">3. Ignores interactions</b><br>
            Education × Income × Family together is far more predictive than any feature alone. 
            Pearson treats each feature in isolation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Method Comparison", "🏆 Composite Ranking",
        "🕸️ Radar Profile", "🔬 Deep Dive"
    ])

    with tab1:
        st.plotly_chart(method_comparison_grouped(result_df), use_container_width=True)
        st.plotly_chart(pearson_vs_truth_divergence(result_df), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="insight-card warning">
                <h4>🔴 CD Account — Most Misleading</h4>
                <p>Pearson ranks it <b>#3</b> but Drop-One AUC shows removing it has <b>zero impact</b> 
                on model performance. Pearson overstates binary features.</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="insight-card positive">
                <h4>🟢 Education — Most Understated</h4>
                <p>Pearson ranks it <b>#5</b> but permutation & drop-one AUC rank it <b>#2–3</b>. 
                Education has a non-linear threshold effect Pearson misses.</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="insight-card info">
                <h4>🔵 Online & CreditCard — True Noise</h4>
                <p>p-values of 0.66 and 0.84 respectively. Both are <b>statistically insignificant</b> 
                and should be excluded from targeted campaigns.</p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.plotly_chart(composite_score_bar(result_df), use_container_width=True)

        st.markdown("##### 📋 Full Verdict Table")
        verdict_df = verdict_summary_table(result_df)
        st.dataframe(
            verdict_df.drop(columns=['Insight']),
            use_container_width=True, hide_index=True,
        )
        st.markdown("##### 💡 Feature-by-Feature Insights")
        for _, row in verdict_df.iterrows():
            verdict, color, desc = row['Verdict'], '#8B949E', row['Insight']
            if '🔥' in verdict: color = '#FF4444'
            elif '⚡' in verdict: color = '#FF9800'
            elif '✅' in verdict: color = '#00C49F'
            elif '🔵' in verdict: color = '#2196F3'
            st.markdown(f"""
            <div style="border-left:3px solid {color};background:#161B22;
                        padding:10px 16px;margin:6px 0;border-radius:0 8px 8px 0;">
                <span style="color:{color};font-weight:700;">#{row['Rank']} {row['Feature']}</span>
                <span style="color:#8B949E;font-size:12px;margin-left:10px;">{row['Verdict']}</span>
                <br><span style="color:#E6EDF3;font-size:13px;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(radar_chart(result_df), use_container_width=True)
        st.markdown("""
        <div class="insight-card info">
            <h4>🕸️ Reading the Radar</h4>
            <p>A feature that scores high on ALL 6 axes is a <b>robust, reliable predictor</b>. 
            <b>Income</b> dominates every axis — it's the only feature that is consistently #1 
            regardless of which method you use. <b>Education</b> and <b>CCAvg</b> show strong but 
            uneven profiles — strong in tree-based methods, weaker in linear ones, confirming 
            their non-linear nature.</p>
        </div>""", unsafe_allow_html=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(drop_one_waterfall(raw_df, baseline_auc), use_container_width=True)
        with col2:
            st.plotly_chart(perm_importance_with_error(raw_df), use_container_width=True)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1a0a,#102d10);border:1px solid #00C49F;
                    border-radius:12px;padding:20px;margin-top:12px;">
            <h4 style="color:#00C49F;margin:0 0 12px 0;">🎯 Final Recommendation: Which Features to Use?</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;color:#E6EDF3;font-size:13px;">
                <div>
                    <b style="color:#FF4444;">✅ KEEP for Modeling (top predictors):</b><br><br>
                    🔥 <b>Income</b> — dominant, irreplaceable<br>
                    ⚡ <b>CCAvg</b> — strong behavioral signal<br>
                    ⚡ <b>Education</b> — non-linear life-stage signal<br>
                    ✅ <b>Family</b> — financial need indicator<br>
                    ✅ <b>CD Account</b> — relationship signal<br>
                    🔵 <b>Mortgage</b> — marginal but retainable<br>
                </div>
                <div>
                    <b style="color:#FF6B6B;">❌ EXCLUDE for Campaign Targeting:</b><br><br>
                    ⚪ <b>Online</b> — p=0.66, true noise<br>
                    ⚪ <b>CreditCard</b> — p=0.84, misleading<br>
                    ⚪ <b>Securities Account</b> — p=0.12, not significant<br>
                    ⚪ <b>Experience</b> — 100% collinear with Age<br>
                    ⚪ <b>Age</b> — non-linear, better captured by Income+Education<br>
                    <br><b style="color:#FF9800;">Baseline AUC with all features: {baseline_auc:.4f}</b>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─── PAGE: PREDICTIVE ─────────────────────────────────────────────────────────
elif page == "🤖  Predictive Modeling":
    st.markdown("""
    <div class="section-header">
        <h3>🤖 Predictive Modeling</h3>
        <p>ML models trained to predict loan interest — evaluate, compare, and predict new customers</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🎯 Predict Customer", "📉 Model Deep Dive"])

    with tab1:
        st.markdown(f"""
        <div class="insight-card positive">
            <h4>🏆 Best Performing Model: {best_model_name}</h4>
            <p>ROC AUC: <b>{results[best_model_name]['roc_auc']:.4f}</b> &nbsp;|&nbsp; 
               Accuracy: <b>{results[best_model_name]['accuracy']*100:.2f}%</b> &nbsp;|&nbsp; 
               F1 Score: <b>{results[best_model_name]['f1']:.4f}</b></p>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(model_comparison_chart(results), use_container_width=True)
        st.plotly_chart(roc_curves_chart(results), use_container_width=True)

        st.markdown("##### 📋 Detailed Model Metrics")
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [f"{results[k]['accuracy']*100:.2f}%" for k in results],
            'F1 Score': [f"{results[k]['f1']:.4f}" for k in results],
            'ROC AUC': [f"{results[k]['roc_auc']:.4f}" for k in results],
            'Precision (Loan=1)': [f"{results[k]['report']['1']['precision']:.4f}" for k in results],
            'Recall (Loan=1)': [f"{results[k]['report']['1']['recall']:.4f}" for k in results],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 🎯 Predict a New Customer's Loan Interest")
        st.markdown("<p style='color:#8B949E;'>Fill in the customer's details below to get a prediction and probability score.</p>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 20, 70, 35)
            income = st.slider("Annual Income ($000)", 8, 250, 80)
            experience = st.slider("Work Experience (years)", 0, 45, 10)
            family = st.selectbox("Family Size", [1, 2, 3, 4], index=1)
        with c2:
            ccavg = st.slider("Avg CC Spending ($/000 per month)", 0.0, 15.0, 2.0, step=0.1)
            mortgage = st.slider("Mortgage ($000)", 0, 700, 0, step=10)
            education = st.selectbox("Education Level", [1, 2, 3],
                                      format_func=lambda x: {1:'Undergrad',2:'Graduate',3:'Advanced/Professional'}[x])
        with c3:
            securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cd_account = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
            online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            credit_card = st.selectbox("UniversalBank Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

        model_choice = st.selectbox("Model to Use", list(results.keys()), index=list(results.keys()).index(best_model_name))

        if st.button("🔮 Predict Loan Interest", use_container_width=True):
            customer = {
                'Age': age, 'Experience': experience, 'Income': income,
                'Family': family, 'CCAvg': ccavg, 'Education': education,
                'Mortgage': mortgage, 'Securities Account': securities,
                'CD Account': cd_account, 'Online': online, 'CreditCard': credit_card,
            }
            pred, prob = predict_customer(customer, results, model_choice)

            col_g, col_r = st.columns([1, 1])
            with col_g:
                st.plotly_chart(prediction_probability_gauge(prob), use_container_width=True)
            with col_r:
                if pred == 1:
                    st.markdown(f"""
                    <div class="pred-yes">
                        <div style="font-size:48px;">✅</div>
                        <div style="font-size:22px; font-weight:700; color:#00C49F; margin:8px 0;">
                            Likely Interested!</div>
                        <div style="color:#8B949E; font-size:14px;">
                            Probability: <b style="color:#00C49F;">{prob*100:.1f}%</b><br>
                            Recommended Action: <b>Send Personalized Offer</b>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-no">
                        <div style="font-size:48px;">❌</div>
                        <div style="font-size:22px; font-weight:700; color:#FF6B6B; margin:8px 0;">
                            Low Interest</div>
                        <div style="color:#8B949E; font-size:14px;">
                            Probability: <b style="color:#FF6B6B;">{prob*100:.1f}%</b><br>
                            Recommended Action: <b>Nurture Relationship</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.session_state['last_customer'] = customer
            st.session_state['last_pred'] = pred
            st.session_state['last_prob'] = prob

            if pred == 1:
                st.success("➡️ Head to **Prescriptive & Messaging** tab to generate a personalized outreach message!")

    with tab3:
        selected_model = st.selectbox("Select Model", list(results.keys()), key='deep_dive')
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(confusion_matrix_chart(results, selected_model), use_container_width=True)
        with col2:
            st.plotly_chart(feature_importance_chart(results, selected_model), use_container_width=True)

# ─── PAGE: PRESCRIPTIVE ───────────────────────────────────────────────────────
elif page == "💬  Prescriptive & Messaging":
    st.markdown("""
    <div class="section-header">
        <h3>💬 Prescriptive Analysis & AI Messaging</h3>
        <p>What should we DO? — Target the right segments and craft the right message</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Target Segments", "📊 Campaign Optimizer", "✉️ Generate Message"])

    with tab1:
        st.plotly_chart(high_value_segments_chart(df), use_container_width=True)
        st.markdown("""
        <div class="insight-card positive">
            <h4>💡 How to Use This</h4>
            <p>Larger boxes = more customers in that segment. Darker red = higher loan acceptance rate. 
            Prioritize the <b>Graduate + 100-150K</b> and <b>Advanced/Prof + 150-200K</b> segments 
            for maximum campaign ROI.</p>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(optimal_threshold_chart(results, best_model_name), use_container_width=True)

    with tab2:
        st.plotly_chart(campaign_roi_simulator(df, results, best_model_name), use_container_width=True)
        st.markdown("""
        <div class="insight-card warning">
            <h4>⚙️ Setting the Right Threshold</h4>
            <p>A lower threshold (e.g. 0.3) casts a wider net — more customers targeted but lower precision. 
            A higher threshold (e.g. 0.6) is more precise but misses potential leads. 
            The optimal sweet spot for this dataset is around <b>0.4–0.5</b> for balanced precision/recall.</p>
        </div>""", unsafe_allow_html=True)

        # High-priority list
        y_prob_all = results[best_model_name]['model'].predict_proba(X)[:, 1]
        df_scored = df.copy()
        df_scored['Loan_Probability'] = y_prob_all
        df_scored['Recommendation'] = df_scored['Loan_Probability'].apply(
            lambda x: '🔥 High Priority' if x >= 0.7 else ('⚡ Medium Priority' if x >= 0.4 else '💤 Low Priority'))
        threshold = st.slider("Show customers above probability threshold:", 0.3, 0.9, 0.6, 0.05)
        top_targets = df_scored[df_scored['Loan_Probability'] >= threshold].sort_values('Loan_Probability', ascending=False)
        st.markdown(f"**{len(top_targets):,} customers** above {threshold*100:.0f}% threshold")
        display_cols = ['Age', 'Income', 'Education_Label', 'Family', 'CCAvg', 'Mortgage',
                        'CD Account', 'Loan_Probability', 'Recommendation']
        st.dataframe(
            top_targets[display_cols].head(20).rename(columns={'Education_Label': 'Education', 'Loan_Probability': 'Probability'}).assign(
                Probability=lambda d: d['Probability'].apply(lambda x: f"{x*100:.1f}%")
            ),
            use_container_width=True, hide_index=True,
        )

    with tab3:
        st.markdown("### ✉️ AI-Powered Personalized Message Generator")
        st.markdown("<p style='color:#8B949E;'>Generate a Claude-powered, hyper-personalized loan outreach message for any customer profile.</p>", unsafe_allow_html=True)

        use_last = False
        if 'last_customer' in st.session_state and st.session_state.get('last_pred') == 1:
            use_last = st.checkbox("✅ Use last predicted customer (from Predictive tab)", value=True)

        if use_last and 'last_customer' in st.session_state:
            customer = st.session_state['last_customer']
            prob = st.session_state['last_prob']
            st.markdown("""
            <div class="insight-card positive">
                <h4>📋 Customer Profile Loaded</h4>
                <p>Using the customer from the Predictive tab. Review below or customize.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("##### 👤 Enter Customer Profile")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 20, 70, 38, key='msg_age')
                income = st.slider("Annual Income ($000)", 8, 250, 120, key='msg_income')
                family = st.selectbox("Family Size", [1,2,3,4], index=2, key='msg_family')
            with c2:
                education = st.selectbox("Education", [1,2,3], index=2,
                                          format_func=lambda x: {1:'Undergrad',2:'Graduate',3:'Advanced/Professional'}[x],
                                          key='msg_edu')
                ccavg = st.slider("CC Spending ($/000/mo)", 0.0, 15.0, 3.5, key='msg_cc')
                mortgage = st.slider("Mortgage ($000)", 0, 700, 150, key='msg_mortgage')
            with c3:
                cd_account = st.selectbox("CD Account", [0,1], format_func=lambda x: "Yes" if x else "No", key='msg_cd')
                securities = st.selectbox("Securities Account", [0,1], format_func=lambda x: "Yes" if x else "No", key='msg_sec')
                online = st.selectbox("Online Banking", [0,1], format_func=lambda x: "Yes" if x else "No", key='msg_online')
                credit_card = st.selectbox("Credit Card", [0,1], format_func=lambda x: "Yes" if x else "No", key='msg_cc2')
            customer = {
                'Age': age, 'Experience': age-22, 'Income': income, 'Family': family,
                'CCAvg': ccavg, 'Education': education, 'Mortgage': mortgage,
                'Securities Account': securities, 'CD Account': cd_account,
                'Online': online, 'CreditCard': credit_card,
            }
            pred, prob = predict_customer(customer, results, best_model_name)

        # Profile summary
        edu_label = {1:'Undergrad', 2:'Graduate', 3:'Advanced/Professional'}.get(customer.get('Education',1))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age", customer.get('Age', '-'))
        m2.metric("Income", f"${customer.get('Income',0)}K")
        m3.metric("Education", edu_label)
        m4.metric("Loan Probability", f"{prob*100:.1f}%")

        if st.button("✨ Generate AI Personalized Message", use_container_width=True):
            if not api_key:
                st.warning("⚠️ No API key provided — using template message. Add your Anthropic API key in the sidebar for AI-generated messages.")
            with st.spinner("🤖 Claude is crafting a personalized message..."):
                message = generate_personalized_message(customer, prob, api_key if api_key else None)

            st.markdown("##### 📨 Generated Message")
            st.markdown(f'<div class="message-box">{message}</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Message", message, file_name="loan_message.txt",
                                    mime="text/plain", use_container_width=True)
            with col2:
                st.info("💡 Copy this message to your CRM or email platform for outreach.")
