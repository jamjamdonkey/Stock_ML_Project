import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
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
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    param = {
        'C': trial.suggest_float("C", 0.01, 10.0, log=True),
        'penalty': penalty,
        'solver': solver,
        'max_iter': 1000,
        'random_state': 42
    }
    model = LogisticRegression(**param)
    score = cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc')
    return score.mean()

# ✅ 5. 튜닝 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# ✅ 6. 결과 출력
print("🎯 Best ROC AUC:", study.best_value)
print("🔧 Best Params:")
for k, v in study.best_params.items():
    print(f" - {k}: {v}")

# ✅ 7. penalty에 맞는 solver 재지정
penalty = study.best_params["penalty"]
solver = "liblinear" if penalty == "l1" else "lbfgs"

# ✅ 8. 최적 모델 학습 및 저장
best_model = LogisticRegression(
    penalty=penalty,
    C=study.best_params["C"],
    solver=solver,
    max_iter=1000,
    random_state=42
)
best_model.fit(X_scaled, y)

joblib.dump(best_model, "best_log_model.pkl")
print("✅ 로지스틱 회귀 최적 모델 저장 완료: best_log_model.pkl")
