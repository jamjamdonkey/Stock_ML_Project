import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ✅ [1] 데이터 불러오기 및 정렬
df = pd.read_csv("Supervised_Learning_CSV/Stock_Screener_FeatureSet_3D.csv", encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by='날짜')

# ✅ [2] 피처, 타겟 분리
drop_cols = ['날짜', '종목코드', '종가', 'target', 'future_return_3d']
X_all = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_all = df['target'].reset_index(drop=True)
ret_all = df['future_return_3d'].reset_index(drop=True)
code_all = df['종목코드'].reset_index(drop=True)

# ✅ [3] Rolling 실험 파라미터 설정
window_size = 80    # 학습기간
test_size = 20      # 테스트기간
step_size = 10      # 이동 간격
topN_list = [1, 3, 5]  # 추천 종목 수

results = []

# ✅ [4] Rolling Loop 시작
for start in range(0, len(df) - window_size - test_size + 1, step_size):
    train_idx = slice(start, start + window_size)
    test_idx = slice(start + window_size, start + window_size + test_size)

    X_train = X_all.iloc[train_idx]
    y_train = y_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    y_test = y_all.iloc[test_idx]
    ret_test = ret_all.iloc[test_idx]
    code_test = code_all.iloc[test_idx]

    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 정의 (LGBM + XGB + LR + SVM)
    model_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                              use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # 앙상블 구성
    ensemble_model = VotingClassifier(
        estimators=[
            ('lgbm', model_lgbm),
            ('xgb', model_xgb),
            ('lr', model_lr),
            ('svm', model_svm)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_scaled, y_train)
    proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]

    # 예측 결과 저장
    test_df = pd.DataFrame({
        '종목코드': code_test.values,
        '예측확률': proba,
        '실제상승': y_test.values,
        '3일후수익률': ret_test.values
    })

    # 회차별 결과 기록
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    result = {'start_index': start, 'roc_auc': roc_auc, 'pr_auc': pr_auc}

    for N in topN_list:
        top_n = test_df.sort_values(by='예측확률', ascending=False).head(N)
        result[f'top{N}_ret'] = top_n['3일후수익률'].mean()
        result[f'top{N}_acc'] = (top_n['실제상승'] == 1).mean()

    results.append(result)

# ✅ [5] 전체 결과 요약 출력
result_df = pd.DataFrame(results)
print("\n✅ Rolling Window 실험 요약 (SVM 포함 4-모델 앙상블):")
for N in topN_list:
    print(f"Top {N}: 평균 수익률={result_df[f'top{N}_ret'].mean():.4f}, 정답률={result_df[f'top{N}_acc'].mean():.4f}")
print(f"전체 평균 ROC AUC: {result_df['roc_auc'].mean():.4f}")
print(f"전체 평균 PR AUC : {result_df['pr_auc'].mean():.4f}")
