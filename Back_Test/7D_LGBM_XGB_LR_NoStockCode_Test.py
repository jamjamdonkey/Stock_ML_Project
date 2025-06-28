import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. 데이터 불러오기
# ======================
df = pd.read_csv("Supervised_Learning_CSV/Stock_Screener_FeatureSet_1D.csv", encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by=['종목코드', '날짜'])

# ======================
# 2. 분포 점검
# ======================
print("🎯 target 분포 (비율):")
print(df['target'].value_counts(normalize=True))

plt.figure(figsize=(12, 6))
sns.histplot(df['future_return_1d'], bins=60, kde=True, color="orange", edgecolor="black")
plt.title("하루 뒤 수익률 분포")
plt.xlabel("수익률 (%)")
plt.ylabel("빈도")
plt.grid(True)
plt.show()

# ======================
# 3. 백테스트 시작
# ======================
returns = []
accuracies = []

unique_dates = sorted(df['날짜'].unique())

for i in tqdm(range(7, len(unique_dates)-3)):  # 마지막 3일 제외 (미래 수익률)
    train_dates = unique_dates[i-7:i]
    test_date = unique_dates[i]

    train = df[df['날짜'].isin(train_dates)]
    test = df[df['날짜'] == test_date]

    # 추후 복원을 위해 종목코드, 수익률 등 저장
    test_codes = test['종목코드'].values
    test_returns = test['future_return_1d'].values
    test_targets = test['target'].values

    drop_cols = ['날짜', '종목코드', '종가', 'target', 'future_return_1d']
    X_train = train.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_train = train['target']
    X_test = test.drop(columns=drop_cols).select_dtypes(include=[np.number])

    # ======================
    # 4. 정규화
    # ======================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ======================
    # 5. 모델 학습
    # ======================
    model_lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    model_xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_lr = LogisticRegression(max_iter=1000, random_state=42)

    model_lgbm.fit(X_train_scaled, y_train)
    model_xgb.fit(X_train_scaled, y_train)
    model_lr.fit(X_train_scaled, y_train)

    # ======================
    # 6. 소프트 보팅 예측
    # ======================
    proba_lgbm = model_lgbm.predict_proba(X_test_scaled)[:, 1]
    proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
    avg_proba = (proba_lgbm + proba_xgb + proba_lr) / 3

    # ======================
    # 7. 종목코드 복원 및 추천 평가
    # ======================
    test_result = pd.DataFrame({
        '종목코드': test_codes,
        'target': test_targets,
        'future_return_1d': test_returns,
        'avg_proba': avg_proba
    })

    # Top-N 추천
    N = 1
    top_n = test_result.sort_values('avg_proba', ascending=False).head(N)

    avg_return = top_n['future_return_1d'].mean()
    acc = (top_n['target'] == 1).mean()

    returns.append(avg_return)
    accuracies.append(acc)

# ======================
# 8. 백테스트 결과 출력
# ======================
print(f"\n✅ 종목코드 분리 + LGBM + XGB + LR Soft Voting 백테스트 결과:")
print(f"📊 평균 수익률: {np.mean(returns):.4f}")
print(f"🎯 평균 정답률: {np.mean(accuracies):.4f}")
