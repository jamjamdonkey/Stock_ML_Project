import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기
df = pd.read_csv('Supervised_Learning_CSV/Train_7days_K4.csv')  # 또는 'Train_5days_K4.csv'로 변경 가능

# 클러스터 라벨이 있는 경우, 원-핫 인코딩 (범주형 처리)
if '클러스터' in df.columns:
    df = pd.get_dummies(df, columns=['클러스터'], drop_first=True)  # drop_first=True로 더미 변수 개수 줄임

# X, y 분리
X = df.drop(columns=['종목코드', '끝날짜', 'X8'])  # X8 = 다음날 종가
y = df['X8']

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.4f}')
