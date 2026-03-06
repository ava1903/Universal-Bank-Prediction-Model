import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data, get_features_target
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from scipy import stats

def compute_all_importances():
    df = load_data()
    X, y = get_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 1. Pearson Correlation
    corr = X.corrwith(y).abs()

    # 2. Mutual Information
    mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)

    # 3. RF Importance
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)

    # 4. Permutation Importance
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_imp = pd.Series(perm.importances_mean, index=X.columns)
    perm_std = pd.Series(perm.importances_std, index=X.columns)

    # 5. Drop-one AUC Impact
    baseline_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    drop_impact = {}
    for col in X.columns:
        m = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        m.fit(X_train.drop(columns=[col]), y_train)
        auc = roc_auc_score(y_test, m.predict_proba(X_test.drop(columns=[col]))[:, 1])
        drop_impact[col] = baseline_auc - auc
    drop_series = pd.Series(drop_impact)

    # 6. LR Coefficients
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(sc.fit_transform(X_train), y_train)
    lr_coef = pd.Series(np.abs(lr.coef_[0]), index=X.columns)

    # Normalize all to 0-1 for radar
    def norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn: return s * 0
        return (s - mn) / (mx - mn)

    result = pd.DataFrame({
        'Pearson Corr': norm(corr),
        'Mutual Info': norm(mi),
        'RF Importance': norm(rf_imp),
        'Permutation': norm(perm_imp.clip(lower=0)),
        'Drop-One AUC': norm(drop_series.clip(lower=0)),
        'LR Coefficient': norm(lr_coef),
    })

    # Composite rank score
    result['Composite Score'] = result.mean(axis=1)
    result['Rank'] = result['Composite Score'].rank(ascending=False).astype(int)
    result = result.sort_values('Composite Score', ascending=False)

    raw = pd.DataFrame({
        'Pearson Corr (raw)': corr,
        'Mutual Info (raw)': mi,
        'RF Importance (raw)': rf_imp,
        'Permutation (raw)': perm_imp,
        'Perm Std': perm_std,
        'Drop-One AUC (raw)': drop_series,
        'LR Coef (raw)': lr_coef,
    })

    return result, raw, baseline_auc

if __name__ == '__main__':
    result, raw, auc = compute_all_importances()
    print(result[['Composite Score', 'Rank']].to_string())
