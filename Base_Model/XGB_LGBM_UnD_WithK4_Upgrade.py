import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 💡 파생 지표 생성 함수
def add_features(df):
    df['range_pct'] = (df['고가'] - df['저가']) / df['시가']
    df['close_vs_open'] = (df['종가'] - df['시가']) / df['시가']
    df['tail_up'] = (df['고가'] - df['종가']) / df['고가']
    df['tail_down'] = (df['종가'] - df['저가']) / df['저가']
    df['volume_price'] = df['종가'] * df['거래량']
    return df

# 1. 데이터 불러오기
df = pd.read_csv('Supervised_Learning_CSV/Train_7days_K4.csv')  # 라벨 포함된 CSV

# 2. 파생 지표 추가
df = add_features(df)

# 3. 상승/하락 라벨 생성
df['label'] = (df['X8'] > 0).astype(int)

# 4. X, y 분리
X = df.drop(columns=['종목코드', '끝날짜', 'X8', 'label'])  # 'cluster'는 포함됨
y = df['label']

# 5. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 모델 정의
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm_clf = LGBMClassifier(random_state=42)

# 7. 앙상블 (Soft Voting)
ensemble = VotingClassifier(
    estimators=[('xgb', xgb_clf), ('lgbm', lgbm_clf)],
    voting='soft'
)

# 8. 모델 훈련
ensemble.fit(X_train, y_train)

# 9. 예측
y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)[:, 1]

# 🔎 평가 지표 출력
print("📊 Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("🎯 Precision:", round(precision_score(y_test, y_pred), 4))
print("📈 Recall:   ", round(recall_score(y_test, y_pred), 4))
print("📌 F1 Score: ", round(f1_score(y_test, y_pred), 4))
print("🚀 ROC AUC:  ", round(roc_auc_score(y_test, y_prob), 4))
