import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import shap

# ✅ [1] 학습 데이터 로드 및 전처리
df = pd.read_csv("Stock_Recomand_Model_V1/Learn.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
df = df.fillna(0)

# ✅ [2] 모델 및 피처 리스트 로딩
model_lgb, feature_cols = joblib.load("Stock_Recomand_Model_V1/Tuning/best_lgbm_model.pkl")
model_xgb = joblib.load("Stock_Recomand_Model_V1/Tuning/best_xgb_model.pkl")
model_svm = joblib.load("Stock_Recomand_Model_V1/Tuning/best_svm_model.pkl")
model_log = joblib.load("Stock_Recomand_Model_V1/Tuning/best_log_model.pkl")

# ✅ [3] 입력 데이터 로드 및 전처리
input_df = pd.read_csv("Stock_Recomand_Model_V1/Input.csv")
input_df.columns = input_df.columns.str.strip().str.lower()
input_df = input_df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
input_df = input_df.fillna(0)

# ✅ [4] LGBM이 요구하는 피처 누락 시 0으로 채움
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# ✅ [5] 예측용 입력 준비 (LGBM, XGB 등)
X_input = input_df[feature_cols].copy()
X_train = df[feature_cols].copy()
y_train = df['target']

# ✅ [6] SHAP용 피처 준비 (미래 정보 제거)
shap_cols = [col for col in feature_cols if col != 'future_return_3d']
X_shap_train = df[shap_cols].copy()
X_shap_input = input_df[shap_cols].copy()

# ✅ [7] 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_input_scaled = scaler.transform(X_input)
X_shap_train_scaled = scaler.fit_transform(X_shap_train)
X_shap_input_scaled = scaler.transform(X_shap_input)

# ✅ [8] 예측 확률 계산
pred_input = {
    'lgb': model_lgb.predict_proba(X_input_scaled)[:, 1],
    'xgb': model_xgb.predict_proba(X_input_scaled)[:, 1],
    'svm': model_svm.predict_proba(X_input_scaled)[:, 1],
    'log': model_log.predict_proba(X_input_scaled)[:, 1]
}

# ✅ [9] 고정된 최적 가중치 적용
opt_weights = np.array([0.20288148, 0.31358469, 0.3049393, 0.17859453])
input_df['예측확률'] = (
    opt_weights[0] * pred_input['lgb'] +
    opt_weights[1] * pred_input['xgb'] +
    opt_weights[2] * pred_input['svm'] +
    opt_weights[3] * pred_input['log']
)

# ✅ [10] 상위 종목 선택
top_df = input_df.sort_values(by='예측확률', ascending=False).head(5).reset_index(drop=True)

# ✅ [11] SHAP 해석 (미래 피처 제외)
explainer = shap.Explainer(model_lgb, X_shap_train_scaled)
shap_values = explainer(X_shap_input_scaled)

# ✅ [12] 최종 출력
for i, row in top_df.iterrows():
    print(f"\n🔹 TOP{i+1} 추천 종목: {int(row['종목코드'])}")
    print(f"   - 예측 확률: {row['예측확률']:.4f}")
    print(f"   - SHAP 기반 추천 사유:")

    idx = row.name
    shap_row = shap_values[idx]
    impact = list(zip(shap_cols, shap_row.values, X_shap_input.iloc[idx].values))
    impact_sorted = sorted(impact, key=lambda x: abs(x[1]), reverse=True)[:3]

    for j, (feat, shap_val, feat_val) in enumerate(impact_sorted, 1):
        print(f"     {j}. {feat}: {feat_val:.4f} (영향력 {'+' if shap_val >= 0 else ''}{shap_val:.4f})")

print("\n✅ 최종 예측 완료 및 SHAP 기반 설명 제공됨")
