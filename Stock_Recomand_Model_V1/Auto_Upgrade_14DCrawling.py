import os
import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

# 현재 파일 기준 절대 경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 저장 경로 설정
learn_path = os.path.join(BASE_DIR, "Learn_14d.csv")
input_path = os.path.join(BASE_DIR, "Input_14d.csv")

# KOSPI 시가총액 상위 50개 종목 코드
stock_codes = [
    "005930", "000660", "207940", "373220", "035420", "034020", "105560", "012450", "005380", "329180",
    "005935", "000270", "068270", "035720", "055550", "009540", "028260", "042660", "012330", "032830",
    "402340", "086790", "011200", "015760", "064350", "005490", "138040", "000810", "323410", "267260",
    "259960", "010130", "316140", "096770", "033780", "010140", "018260", "051910", "034730", "024110",
    "006400", "003550", "030200", "352820", "006800", "066570", "377300", "017670", "079550", "272210"
]

# 주가 데이터 크롤링 함수
def get_recent_prices(code, days=120):
    base_url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df_total = pd.DataFrame()
    page = 1

    while len(df_total) < days:
        url = f"{base_url}&page={page}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="type2")

        try:
            df = pd.read_html(str(table))[0]
        except:
            break

        df = df.dropna()
        df.columns = ["날짜", "종가", "전일비", "시가", "고가", "저가", "거래량"]
        df["종목코드"] = code
        df_total = pd.concat([df_total, df], ignore_index=True)

        page += 1
        time.sleep(0.3)

    df_total["날짜"] = pd.to_datetime(df_total["날짜"])
    df_total = df_total.sort_values("날짜").tail(days).reset_index(drop=True)
    return df_total

# 단일 행에 대한 파생 피처 생성
def add_single_day_features(row):
    row['range_pct'] = (row['고가'] - row['저가']) / row['시가'] if row['시가'] != 0 else 0
    row['close_vs_open'] = (row['종가'] - row['시가']) / row['시가'] if row['시가'] != 0 else 0
    row['tail_up'] = (row['고가'] - row['종가']) / row['고가'] if row['고가'] != 0 else 0
    row['tail_down'] = (row['종가'] - row['저가']) / row['저가'] if row['저가'] != 0 else 0
    row['volume_price'] = row['종가'] * row['거래량']
    return row

# 학습용 Learn_14d.csv 생성
def generate_training_set(recent_data, output_path):
    feature_rows = []

    for code in stock_codes:
        df = recent_data.get(code)
        if df is None or len(df) < 30:
            continue

        df = df.sort_values("날짜").reset_index(drop=True)
        df = df.apply(add_single_day_features, axis=1)

        for i in range(len(df) - 28):  # 14 + 14
            window = df.iloc[i:i+14]
            future = df.iloc[i+14:i+28]

            closes = window['종가'].values
            future_close = future['종가'].values

            if len(future_close) < 14:
                continue

            return_14d = (future_close[-1] - closes[-1]) / closes[-1]
            target = 1 if return_14d > 0 else 0

            row = {
                '종목코드': code,
                '날짜': window.iloc[-1]['날짜'],
                'future_return_14d': return_14d,
                'target': target
            }

            for j in range(14):
                row[f'close_{j+1}'] = closes[j]

            row['mean_close'] = np.mean(closes)
            row['std_close'] = np.std(closes)
            row['return_1d'] = (closes[-1] - closes[-2]) / closes[-2]
            row['return_14d'] = (closes[-1] - closes[0]) / closes[0]
            row['num_up_days'] = sum(closes[j] > closes[j-1] for j in range(1, 14))

            last_day_feat = window.iloc[-1].copy()
            row = {**row, **add_single_day_features(last_day_feat)}
            feature_rows.append(row)

    final_df = pd.DataFrame(feature_rows)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"학습용 저장 완료: {output_path} ({len(final_df)} rows)")

# 예측용 Input_14d.csv 생성
def generate_today_input(recent_data, output_path):
    feature_rows = []

    for code in stock_codes:
        df = recent_data.get(code)
        if df is None or len(df) < 14:
            continue

        df = df.sort_values("날짜").reset_index(drop=True)
        df = df.apply(add_single_day_features, axis=1)
        window = df.tail(14)

        closes = window['종가'].values
        if len(closes) < 14:
            continue

        row = {
            '종목코드': code,
            '날짜': window.iloc[-1]['날짜']
        }

        for j in range(14):
            row[f'close_{j+1}'] = closes[j]

        row['mean_close'] = np.mean(closes)
        row['std_close'] = np.std(closes)
        row['return_1d'] = (closes[-1] - closes[-2]) / closes[-2]
        row['return_14d'] = (closes[-1] - closes[0]) / closes[0]
        row['num_up_days'] = sum(closes[j] > closes[j-1] for j in range(1, 14))

        last_day_feat = window.iloc[-1].copy()
        row = {**row, **add_single_day_features(last_day_feat)}
        feature_rows.append(row)

    final_df = pd.DataFrame(feature_rows)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"입력용 저장 완료: {output_path} ({len(final_df)} rows)")

# 메인 실행
if __name__ == "__main__":
    print("전체 종목 데이터 수집 중...")
    recent_data_dict = {}
    for code in tqdm(stock_codes):
        try:
            recent_data_dict[code] = get_recent_prices(code, days=120)
        except Exception as e:
            print(f"오류 {code}: {e}")

    recent_data_train = {code: df.iloc[:92].copy() for code, df in recent_data_dict.items() if len(df) >= 120}
    recent_data_input = {code: df.iloc[92:].copy() for code, df in recent_data_dict.items() if len(df) >= 120}

    generate_training_set(recent_data_train, learn_path)
    generate_today_input(recent_data_input, input_path)
