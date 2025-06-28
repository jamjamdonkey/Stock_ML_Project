import pandas as pd
import os

# CSV들이 들어있는 폴더 경로
folder_path = r'C:\Users\JAMJAM\Stock_Project\Stock_Project\CSV_By_Date'

# CSV 파일 리스트 가져오기 (확장자 필터링)
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 전체 통합 데이터프레임
df_total = pd.DataFrame()

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    
    # 필수 컬럼 검사
    required_cols = {'날짜', '종목코드', '종가', '전일비', '시가', '고가', '저가', '거래량'}
    if not required_cols.issubset(df.columns):
        print(f"[경고] {file_name} 누락 컬럼 있음 → 스킵됨")
        continue

    df_total = pd.concat([df_total, df], ignore_index=True)

# 날짜 형식 정렬
df_total['날짜'] = pd.to_datetime(df_total['날짜'])
df_total = df_total.sort_values(by=['종목코드', '날짜'])

# 저장
save_path = r'C:\Users\JAMJAM\Stock_Project\3Y_ Merged_Stock_Data.csv'
df_total.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"✅ 통합 완료! 저장 위치: {save_path}")
