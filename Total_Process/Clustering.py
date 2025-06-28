import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ 상대경로 기반 파일 불러오기
df = pd.read_csv('Total_Process/3Y_Merged_Stock_Data.csv', encoding="utf-8-sig")

# ✅ 날짜 정렬
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(['종목코드', '날짜'])

# ✅ 3일 수익률 계산
df['수익률_3d'] = df.groupby('종목코드')['종가'].pct_change(periods=3)

# ✅ 종목코드별로 특징 추출
feature_df = df.groupby('종목코드').agg({
    '수익률_3d': ['mean', 'std'],
    '거래량': 'mean',
    '종가': ['mean', 'std'],
    '시가': 'mean',
    '고가': 'mean',
    '저가': 'mean',
}).dropna()

# ✅ 다중 인덱스 정리
feature_df.columns = ['_'.join(col) for col in feature_df.columns]
feature_df = feature_df.reset_index()

# ✅ 정규화
X = feature_df.drop(columns='종목코드')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)
feature_df['클러스터'] = kmeans.fit_predict(X_scaled)

# ✅ 결과 저장 (상대 경로)
save_dir = "Total_Process"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "Stock_Clusters.csv")
feature_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# ✅ 콘솔 출력
print("📊 클러스터링 결과 요약 (상위 5개):")
print(feature_df.head())

print("\n📈 클러스터별 종목 수:")
print(feature_df['클러스터'].value_counts().sort_index())

print("\n✅ 저장된 CSV 경로:")
print(output_path)
