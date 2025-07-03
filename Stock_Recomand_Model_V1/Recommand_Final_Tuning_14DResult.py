import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import os

from Stock_Name import stock_name_dict  # ✅ 종목명 딕셔너리 추가

# 절대경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
learn_path = os.path.join(BASE_DIR, "Learn_14D.csv")  # ✅ 14일 데이터 경로
input_path = os.path.join(BASE_DIR, "Input_14D.csv")
output_path = os.path.join(BASE_DIR, "Top10_Recommanded_14D.csv")

# [1] 학습 데이터 불러오기
df = pd.read_csv(learn_path, encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by='날짜')

# [2] 피처 및 타겟 정의
drop_cols = ['날짜', '종목코드', '종가', 'target', 'future_return_14d']  # ✅ 컬럼명 변경
X_train = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_train = df['target']
train_features = X_train.columns.tolist()

# [3] 입력 데이터 불러오기
input_df = pd.read_csv(input_path, encoding='utf-8-sig')
code_input = input_df['종목코드'].astype(str).str.zfill(6).values
X_input = input_df[train_features].copy()

# [4] 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_input_scaled = scaler.transform(X_input)

# [5] 모델 정의 및 학습
model_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)

model_lgbm.fit(X_train_scaled, y_train)
model_xgb.fit(X_train_scaled, y_train)
model_lr.fit(X_train_scaled, y_train)
model_svm.fit(X_train_scaled, y_train)

# [6] 예측 확률 계산
pred_lgbm = model_lgbm.predict_proba(X_input_scaled)[:, 1]
pred_xgb  = model_xgb.predict_proba(X_input_scaled)[:, 1]
pred_svm  = model_svm.predict_proba(X_input_scaled)[:, 1]
pred_lr   = model_lr.predict_proba(X_input_scaled)[:, 1]

# [7] 가중치 적용
opt_weights = [0.29277898, 0.42287964, 0.15808862, 0.12625276]
final_score = (
    opt_weights[0] * pred_lgbm +
    opt_weights[1] * pred_xgb +
    opt_weights[2] * pred_svm +
    opt_weights[3] * pred_lr
)

input_df['예측확률'] = final_score
input_df['종목코드'] = input_df['종목코드'].astype(str).str.zfill(6)
input_df['종목명'] = input_df['종목코드'].map(stock_name_dict)

top_df = input_df.sort_values(by='예측확률', ascending=False).head(10).reset_index(drop=True)

# [8] SHAP 설명 (LGBM 기준)
explainer = shap.Explainer(model_lgbm, X_train_scaled)
shap_values = explainer(X_input_scaled)

# [9] 출력
for i, row in top_df.iterrows():
    code = row['종목코드']
    name = row['종목명'] if pd.notna(row['종목명']) else "종목명 미상"
    print(f"\nTOP{i+1} 추천 종목: {code} ({name})")
    print(f" - 예측 확률: {row['예측확률']:.4f}")
    print(f" - SHAP 기반 추천 사유:")

    idx = row.name
    shap_row = shap_values[idx]
    impact = list(zip(train_features, shap_row.values, X_input.iloc[idx].values))
    impact_sorted = sorted(impact, key=lambda x: abs(x[1]), reverse=True)[:3]

    for j, (feat, shap_val, feat_val) in enumerate(impact_sorted, 1):
        print(f"   {j}. {feat}: {feat_val:.4f} (영향력 {'+' if shap_val >= 0 else ''}{shap_val:.4f})")

# [10] CSV 저장
shap_features = []

for i, row in top_df.iterrows():
    idx = row.name
    shap_row = shap_values[idx]
    impact = list(zip(train_features, shap_row.values, X_input.iloc[idx].values))
    impact_sorted = sorted(impact, key=lambda x: abs(x[1]), reverse=True)[:3]

    shap_info = {
        '종목코드': row['종목코드'],
        '종목명': row['종목명'],
        '예측확률': row['예측확률']
    }

    for j, (feat, shap_val, feat_val) in enumerate(impact_sorted, 1):
        shap_info[f'SHAP_Feature_{j}'] = feat
        shap_info[f'SHAP_Value_{j}'] = feat_val
        shap_info[f'SHAP_Impact_{j}'] = shap_val

    shap_features.append(shap_info)

shap_df = pd.DataFrame(shap_features)
shap_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("\nTop10_Recommanded_14D.csv 저장 완료 (SHAP 상위 피처 + 종목명 포함)")
