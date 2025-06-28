import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ✅ [1] 데이터 불러오기
df = pd.read_csv("Supervised_Learning_CSV/Stock_Screener_FeatureSet_3D.csv", encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by='날짜')

# ✅ [2] 피처 및 타겟 분리
drop_cols = ['날짜', '종목코드', '종가', 'target', 'future_return_3d']
X_all = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_all = df['target'].reset_index(drop=True)
ret_all = df['future_return_3d'].reset_index(drop=True)
code_all = df['종목코드'].reset_index(drop=True)

# ✅ [3] 실험 설정
window_size = 120
test_size = 20
step_size = 10
topN_list = [1, 3, 5]
threshold = 0.5

results_topN = []
results_thresh = []

# ✅ [4] Rolling Window 실험
for start in range(0, len(df) - window_size - test_size + 1, step_size):
    train_idx = slice(start, start + window_size)
    test_idx = slice(start + window_size, start + window_size + test_size)

    X_train = X_all.iloc[train_idx]
    y_train = y_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    y_test = y_all.iloc[test_idx]
    ret_test = ret_all.iloc[test_idx]
    code_test = code_all.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 앙상블 모델 정의
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
    proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    # 📌 공통 예측 결과
    test_df = pd.DataFrame({
        '종목코드': code_test.values,
        '예측확률': proba,
        '실제상승': y_test.values,
        '3일후수익률': ret_test.values
    })

    # ✅ Top-N 방식
    result_n = {'start_index': start}
    for N in topN_list:
        top_n = test_df.sort_values(by='예측확률', ascending=False).head(N)
        result_n[f'top{N}_ret'] = top_n['3일후수익률'].mean()
        result_n[f'top{N}_acc'] = (top_n['실제상승'] == 1).mean()
    result_n['roc_auc'] = roc_auc_score(y_test, proba)
    result_n['pr_auc'] = average_precision_score(y_test, proba)
    results_topN.append(result_n)

    # ✅ Threshold 방식
    confident = test_df[test_df['예측확률'] > threshold]
    result_t = {
        'start_index': start,
        'n_selected': len(confident),
        'avg_ret': confident['3일후수익률'].mean() if not confident.empty else 0,
        'accuracy': (confident['실제상승'] == 1).mean() if not confident.empty else 0,
        'roc_auc': roc_auc_score(y_test, proba),
        'pr_auc': average_precision_score(y_test, proba)
    }
    results_thresh.append(result_t)

# ✅ [5] 전체 결과 요약 DataFrame 생성
df_topN = pd.DataFrame(results_topN)
df_thresh = pd.DataFrame(results_thresh)

# ✅ [6] 전체 결과 요약 출력
print("\n✅ [Top-N 방식] Rolling Window 실험 요약:")
for N in topN_list:
    avg_ret = df_topN[f'top{N}_ret'].mean()
    avg_acc = df_topN[f'top{N}_acc'].mean()
    print(f"🔹 Top {N} - 평균 수익률: {avg_ret:.4f}, 정답률: {avg_acc:.4f}")

print(f"📊 전체 평균 ROC AUC: {df_topN['roc_auc'].mean():.4f}")
print(f"📊 전체 평균 PR  AUC: {df_topN['pr_auc'].mean():.4f}")

print("\n✅ [Threshold 방식] Rolling Window 실험 요약:")
print(f"🔹 평균 선택 종목 수: {df_thresh['n_selected'].mean():.2f}")
print(f"🔹 평균 수익률: {df_thresh['avg_ret'].mean():.4f}")
print(f"🔹 정답률: {df_thresh['accuracy'].mean():.4f}")
print(f"📊 전체 평균 ROC AUC: {df_thresh['roc_auc'].mean():.4f}")
print(f"📊 전체 평균 PR  AUC: {df_thresh['pr_auc'].mean():.4f}")
