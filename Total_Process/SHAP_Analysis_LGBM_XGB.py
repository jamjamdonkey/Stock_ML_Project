import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# ✅ 한글 폰트 설정 (Windows: 맑은 고딕 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 데이터 불러오기
df = pd.read_csv("Total_Process/Train_7days_K4_Upgrade_3Y_WithCluster.csv", encoding='utf-8-sig')
df = df.dropna()

# ✅ 피처/타겟 분리
X = df.drop(columns=['날짜', '종목코드', 'X8', 'label']).select_dtypes(include=[np.number])
y = df['label']

# ✅ 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ XGBoost 학습 및 SHAP 분석
print("\n📊 XGBoost SHAP 분석 중...")
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_scaled, y)

explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_scaled)

plt.figure()
shap.summary_plot(shap_values_xgb, X, plot_type='violin', show=False)
plt.title("XGBoost - SHAP Summary Plot")
plt.tight_layout()
plt.savefig("Total_Process/SHAP_XGB_Korean.png")
plt.close()

# ✅ LightGBM 학습 및 SHAP 분석
print("\n📊 LightGBM SHAP 분석 중...")
lgbm_model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
lgbm_model.fit(X_scaled, y)

explainer_lgbm = shap.Explainer(lgbm_model)
shap_values_lgbm = explainer_lgbm(X_scaled)

plt.figure()
shap.summary_plot(shap_values_lgbm, X, plot_type='violin', show=False)
plt.title("LGBM - SHAP Summary Plot")
plt.tight_layout()
plt.savefig("Total_Process/SHAP_LGBM_Korean.png")
plt.close()

print("\n✅ SHAP 시각화 완료 및 저장 완료 (파일 저장 위치: Total_Process/)")
