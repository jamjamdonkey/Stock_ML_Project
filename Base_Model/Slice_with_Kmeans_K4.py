import pandas as pd

# 1. 데이터 불러오기
train_5 = pd.read_csv('Supervised_Learning_CSV/Train_5days.csv')
train_7 = pd.read_csv('Supervised_Learning_CSV/Train_7days.csv')
clusters = pd.read_csv('Supervised_Learning_CSV/클러스터링_결과_K4.csv')

# 2. 클러스터링 결과에서 필요한 컬럼만 추출
clusters = clusters[['종목코드', '클러스터']]

# 3. 종목코드를 기준으로 merge
train_5_clustered = train_5.merge(clusters, on='종목코드', how='left')
train_7_clustered = train_7.merge(clusters, on='종목코드', how='left')

# 4. 저장
train_5_clustered.to_csv('Supervised_Learning_CSV/Train_5days_K4.csv', index=False, encoding='utf-8-sig')
train_7_clustered.to_csv('Supervised_Learning_CSV/Train_7days_K4.csv', index=False, encoding='utf-8-sig')

print("✅ 클러스터 라벨 추가 완료: Train_5days_K4.csv, Train_7days_K4.csv")
