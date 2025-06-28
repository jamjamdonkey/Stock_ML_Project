import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 데이터 불러오기
df = pd.read_csv("Supervised_Learning_CSV/Stock_Screener_FeatureSet_1D.csv", encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by=['종목코드', '날짜'])

# 종목코드 기준 학습/테스트 분할
unique_codes = df['종목코드'].unique()
train_codes = unique_codes[:int(len(unique_codes)*0.8)]
test_codes = unique_codes[int(len(unique_codes)*0.8):]

returns = []
accuracies = []

unique_dates = sorted(df['날짜'].unique())

for i in tqdm(range(7, len(unique_dates)-1)):  # 1일 후 수익률 기준
    train_dates = unique_dates[i-7:i]
    test_date = unique_dates[i]

    train = df[(df['날짜'].isin(train_dates)) & (df['종목코드'].isin(train_codes))]
    test = df[(df['날짜'] == test_date) & (df['종목코드'].isin(test_codes))]

    if train.empty or test.empty:
        continue

    drop_cols = ['날짜', '종목코드', '종가', 'target', 'future_return_1d']
    X_train = train.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_train = train['target']
    X_test = test.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_test = test['target']
    test_returns = test['future_return_1d'].values

    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 정의
    model_lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    model_xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_lr = LogisticRegression(max_iter=1000, random_state=42)

    model_lgbm.fit(X_train_scaled, y_train)
    model_xgb.fit(X_train_scaled, y_train)
    model_lr.fit(X_train_scaled, y_train)

    # Soft Voting
    proba_lgbm = model_lgbm.predict_proba(X_test_scaled)[:, 1]
    proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
    avg_proba = (proba_lgbm + proba_xgb + proba_lr) / 3

    test = test.copy()
    test['avg_proba'] = avg_proba
    test['future_return_1d'] = test_returns

    N = 1  # Top N 추천 종목 수
    top_n = test.sort_values('avg_proba', ascending=False).head(N)

    avg_return = top_n['future_return_1d'].mean()
    acc = (top_n['target'] == 1).mean()

    returns.append(avg_return)
    accuracies.append(acc)

# 결과 출력
print(f"\n✅ 종목코드 분리 + LGBM + XGB + LR Soft Voting 백테스트 결과:")
print(f"📊 평균 수익률: {np.mean(returns):.4f}")
print(f"🎯 평균 정답률: {np.mean(accuracies):.4f}")

