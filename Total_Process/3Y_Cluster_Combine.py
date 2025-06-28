import pandas as pd
import os

# 📁 경로 설정
train_path = "Total_Process/Train_7days_K4_Upgrade_3Y.csv"
cluster_path = "Total_Process/Stock_Clusters.csv"
save_path = "Total_Process/Train_7days_K4_Upgrade_3Y_WithCluster.csv"

# ✅ 데이터 불러오기
train_df = pd.read_csv(train_path, encoding='utf-8-sig')
cluster_df = pd.read_csv(cluster_path, encoding='utf-8-sig')

# ✅ 클러스터 컬럼만 추출
cluster_info = cluster_df[['종목코드', '클러스터']]

# ✅ 병합 (종목코드 기준)
merged_df = pd.merge(train_df, cluster_info, on='종목코드', how='left')

# ✅ 저장
merged_df.to_csv(save_path, index=False, encoding='utf-8-sig')
print("✅ 병합 완료! 저장 위치:", save_path)
