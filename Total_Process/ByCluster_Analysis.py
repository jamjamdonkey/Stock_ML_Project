import pandas as pd

# ✅ 데이터 불러오기
df = pd.read_csv("Total_Process/Train_7days_K4_Upgrade_3Y_WithCluster.csv", encoding='utf-8-sig')

# ✅ 결측치 제거
df = df.dropna()

# ✅ 수익률 기준 정의 (3일 후 수익률 1% 초과를 상승으로 정의)
df['is_up'] = (df['X8'] > 0.01).astype(int)

# ✅ 📊 클러스터별 평균 3일 수익률
mean_returns = df.groupby('클러스터')['X8'].mean()
print("📊 클러스터별 평균 3일 수익률:")
print(mean_returns, '\n')

# ✅ 📈 클러스터별 상승 비율 (수익률 > 1%)
up_ratios = df.groupby('클러스터')['is_up'].mean()
print("📈 클러스터별 상승 비율 (수익률 > 1%):")
print(up_ratios, '\n')

# ✅ 📌 클러스터별 샘플 수
sample_counts = df['클러스터'].value_counts().sort_index()
print("📌 클러스터별 샘플 수:")
print(sample_counts, '\n')

# ✅ 🔍 클러스터 3번 종목코드 추출 (중복 제거)
cluster_3_codes = df[df['클러스터'] == 3]['종목코드'].unique()
print(f"🔍 클러스터 3번 종목코드 목록 ({len(cluster_3_codes)}개):")
print(cluster_3_codes)
