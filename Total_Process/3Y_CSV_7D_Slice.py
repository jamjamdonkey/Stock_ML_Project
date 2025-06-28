import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ✅ 파일 불러오기 (상대경로 기준)
df = pd.read_csv("Total_Process/3Y_Merged_Stock_Data.csv", encoding='utf-8-sig')
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(['종목코드', '날짜'])

# ✅ 결과 저장용 리스트
all_data = []

# 🎯 이진 분류 라벨 함수
def classify_label(ret):
    return 1 if ret > 0.01 else 0  # 수익률 1% 초과 상승이면 1

# 📌 당일 기준 파생 피처 생성 함수
def add_single_day_features(row):
    row['range_pct'] = (row['고가'] - row['저가']) / row['시가'] if row['시가'] != 0 else 0
    row['close_vs_open'] = (row['종가'] - row['시가']) / row['시가'] if row['시가'] != 0 else 0
    row['tail_up'] = (row['고가'] - row['종가']) / row['고가'] if row['고가'] != 0 else 0
    row['tail_down'] = (row['종가'] - row['저가']) / row['저가'] if row['저가'] != 0 else 0
    row['volume_price'] = row['종가'] * row['거래량']
    return row

# 🔁 종목별 슬라이딩 윈도우 반복
for code in tqdm(df['종목코드'].unique()):
    stock_df = df[df['종목코드'] == code].reset_index(drop=True)

    for i in range(len(stock_df) - 8):  # 7일 학습 + 1일 라벨
        window = stock_df.iloc[i:i+7]
        next_day = stock_df.iloc[i+7]

        if window.isnull().any().any() or next_day.isnull().any():
            continue

        # 📌 입력 X
        row = {
            '종목코드': code,
            '끝날짜': window.iloc[-1]['날짜'],
        }
        for j in range(7):
            row[f'close_{j+1}'] = window.iloc[j]['종가']

        # 📌 요약 통계 피처
        closes = window['종가'].values
        row['mean_close'] = np.mean(closes)
        row['std_close'] = np.std(closes)
        row['return_1d'] = (closes[-1] - closes[-2]) / closes[-2]
        row['return_3d'] = (closes[-1] - closes[-4]) / closes[-4]
        row['num_up_days'] = sum(closes[j] > closes[j-1] for j in range(1, 7))

        # 📌 단일 일자 파생 피처
        row = add_single_day_features(window.iloc[-1].copy())

        # 📌 출력 Y
        today_close = window.iloc[-1]['종가']
        tomorrow_close = next_day['종가']
        ret = (tomorrow_close - today_close) / today_close
        row['X8'] = ret
        row['label'] = classify_label(ret)

        all_data.append(row)

# ✅ 데이터프레임으로 저장
final_df = pd.DataFrame(all_data)

# 📁 저장 경로 (상대경로)
os.makedirs("Total_Process", exist_ok=True)
final_df.to_csv("Total_Process/Train_7days_K4_Upgrade_3Y.csv", index=False, encoding='utf-8-sig')

print("✅ CSV 생성 완료: Train_7days_K4_Upgrade_3Y.csv")