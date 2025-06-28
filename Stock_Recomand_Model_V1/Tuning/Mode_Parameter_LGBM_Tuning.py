import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ 1. 데이터 불러오기 및 전처리
df = pd.read_csv("Stock_Recomand_Model_V1/Learn.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
df = df.fillna(0)

# ✅ 2. 피처 / 타겟 분리
drop_cols = ['종목코드', '날짜', 'target']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]  # 🔧 DataFrame 유지!
y = df['target']

# ✅ 3. 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Optuna 튜닝 목적함수 정의
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'n_estimators': 100,
        'random_state': 42
    }

    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc')
    return score.mean()

# ✅ 5. 튜닝 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# ✅ 6. 결과 출력
print("🎯 Best ROC AUC:", study.best_value)
print("🔧 Best Params:")
for k, v in study.best_params.items():
    print(f" - {k}: {v}")

# ✅ 7. 최적 모델 학습 및 저장 (feature_cols 함께)
best_model = LGBMClassifier(**study.best_params)
best_model.fit(X_scaled, y)

# 🔧 모델과 feature_cols 함께 저장
joblib.dump((best_model, feature_cols), "Stock_Recomand_Model_V1/best_lgbm_model.pkl")
print("✅ 최적 모델 저장 완료: best_lgbm_model.pkl (모델 + 피처명 포함)")
