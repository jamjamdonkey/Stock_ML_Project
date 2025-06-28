import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

# ✅ 1. 데이터 불러오기 및 전처리
df = pd.read_csv("Stock_Recomand_Model_V1/Learn.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
df = df.fillna(0)

# ✅ 2. 피처 / 타겟 분리
drop_cols = ['종목코드', '날짜', 'target']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]
y = df['target']

# ✅ 3. 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Optuna 목적함수 정의
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'n_estimators': 100,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**param)
    score = cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc')
    return score.mean()

# ✅ 5. 튜닝 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# ✅ 6. 결과 출력 및 저장
print("🎯 Best ROC AUC:", study.best_value)
print("🔧 Best Params:")
for k, v in study.best_params.items():
    print(f" - {k}: {v}")

best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_model.fit(X_scaled, y)

# 저장
joblib.dump(best_model, "best_xgb_model.pkl")
print("✅ XGB 최적 모델 저장 완료: best_xgb_model.pkl")
