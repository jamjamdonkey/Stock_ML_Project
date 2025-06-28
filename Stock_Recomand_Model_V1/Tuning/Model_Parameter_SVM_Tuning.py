import pandas as pd
import numpy as np
import optuna
from sklearn.svm import SVC
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
        'C': trial.suggest_float('C', 0.01, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'kernel': 'rbf',  # SVM은 rbf가 일반적으로 가장 성능 좋음
        'probability': True,
        'random_state': 42
    }
    model = SVC(**param)
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

best_model = SVC(**study.best_params, probability=True, random_state=42)
best_model.fit(X_scaled, y)

# 저장
joblib.dump(best_model, "best_svm_model.pkl")
print("✅ SVM 최적 모델 저장 완료: best_svm_model.pkl")
