import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

# ✅ [1] 학습 데이터 불러오기
df = pd.read_csv("Stock_Recomand_Model_V1/Learn.csv", encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by='날짜')

# ✅ [2] 피처 및 타겟 정의
drop_cols = ['날짜', '종목코드', '종가', 'target', 'future_return_3d']
X_train = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_train = df['target']
train_features = X_train.columns.tolist()

# ✅ [3] 입력 데이터 불러오기
input_df = pd.read_csv("Stock_Recomand_Model_V1/Input.csv", encoding='utf-8-sig')
code_input = input_df['종목코드'].values
X_input = input_df[train_features].copy()

# ✅ [4] 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_input_scaled = scaler.transform(X_input)

# ✅ [5] 모델 구성 및 학습 (앙상블)
model_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)

ensemble = VotingClassifier(
    estimators=[('lgbm', model_lgbm), ('xgb', model_xgb), ('lr', model_lr), ('svm', model_svm)],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)
proba = ensemble.predict_proba(X_input_scaled)[:, 1]

input_df['예측확률'] = proba
top_df = input_df.sort_values(by='예측확률', ascending=False).head(3).reset_index(drop=True)

# ✅ [6] SHAP 기반 설명 (LGBM 단독 학습)
model_lgbm.fit(X_train_scaled, y_train)
explainer = shap.Explainer(model_lgbm, X_train_scaled)
shap_values = explainer(X_input_scaled)

# ✅ [7] 최종 출력
for i, row in top_df.iterrows():
    print(f"\n🔹 TOP{i+1} 추천 종목: {row['종목코드']}")
    print(f"   - 예측 확률: {row['예측확률']:.4f}")
    print(f"   - SHAP 기반 추천 사유:")
    
    idx = row.name
    shap_row = shap_values[idx]
    impact = list(zip(train_features, shap_row.values, X_input.iloc[idx].values))
    impact_sorted = sorted(impact, key=lambda x: abs(x[1]), reverse=True)[:3]

    for j, (feat, shap_val, feat_val) in enumerate(impact_sorted, 1):
        print(f"     {j}. {feat}: {feat_val:.4f} (영향력 {'+' if shap_val >= 0 else ''}{shap_val:.4f})")

print("\n✅ 예측 완료 및 SHAP 기반 상위 3종목 설명 제공됨")
