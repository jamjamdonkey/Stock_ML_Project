import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

# 1. CSV 불러오기
learn_df = pd.read_csv("Stock_Recomand_Model_V1/Learn.csv")
input_df = pd.read_csv("Stock_Recomand_Model_V1/Input.csv")

# 2. 컬럼 전처리
learn_df.columns = learn_df.columns.str.strip().str.lower()
input_df.columns = input_df.columns.str.strip().str.lower()

# 3. object 타입 → 숫자형 변환 + 결측치 처리
learn_df = learn_df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
input_df = input_df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
learn_df = learn_df.fillna(0)
input_df = input_df.fillna(0)

# 4. 피처/타겟 지정
drop_cols = ['종목코드', '날짜', 'target']
common_features = [col for col in learn_df.columns if col not in drop_cols and col in input_df.columns]

X = learn_df[common_features]
y = learn_df['target']

# 5. 모델 학습
model_lgb = LGBMClassifier().fit(X, y)
model_xgb = XGBClassifier().fit(X, y)
model_svm = SVC(probability=True).fit(X, y)
model_log = LogisticRegression(max_iter=1000).fit(X, y)

# 6. 예측 확률 저장 (학습셋)
learn_df['pred_lgb'] = model_lgb.predict_proba(X)[:, 1]
learn_df['pred_xgb'] = model_xgb.predict_proba(X)[:, 1]
learn_df['pred_svm'] = model_svm.predict_proba(X)[:, 1]
learn_df['pred_log'] = model_log.predict_proba(X)[:, 1]

# 7. 최적 가중치 탐색
def objective(weights):
    weights = np.abs(weights)
    weights /= np.sum(weights)
    ensemble = (
        weights[0] * learn_df['pred_lgb'] +
        weights[1] * learn_df['pred_xgb'] +
        weights[2] * learn_df['pred_svm'] +
        weights[3] * learn_df['pred_log']
    )
    return -roc_auc_score(y, ensemble)

init_weights = [0.25, 0.25, 0.25, 0.25]
res = minimize(objective, init_weights, method='Nelder-Mead')
opt_weights = res.x / np.sum(res.x)
print("🔧 최적 가중치:", opt_weights)

# 8. Input 데이터 예측
input_X = input_df[common_features]
input_df['pred_lgb'] = model_lgb.predict_proba(input_X)[:, 1]
input_df['pred_xgb'] = model_xgb.predict_proba(input_X)[:, 1]
input_df['pred_svm'] = model_svm.predict_proba(input_X)[:, 1]
input_df['pred_log'] = model_log.predict_proba(input_X)[:, 1]

# 9. 최종 점수 계산
input_df['final_score'] = (
    opt_weights[0] * input_df['pred_lgb'] +
    opt_weights[1] * input_df['pred_xgb'] +
    opt_weights[2] * input_df['pred_svm'] +
    opt_weights[3] * input_df['pred_log']
)

# 10. 추천 종목 출력
top_n = input_df.sort_values('final_score', ascending=False).head(10)
print("📈 추천 종목 Top 10:")
print(top_n[['종목코드', 'final_score']])
