import pandas as pd

# 1. CSV 불러오기
df = pd.read_csv("Merged_Stock_Data.csv")

# 2. 날짜 변환 및 정렬
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by=['종목코드', '날짜'])

# 3. 전일비 전처리 (부호 적용)
def preprocess_change(change_str):
    if isinstance(change_str, str):
        change_str = change_str.replace(',', '').strip()
        if '하락' in change_str:
            return -float(change_str.replace('하락', '').strip())
        elif '상승' in change_str:
            return float(change_str.replace('상승', '').strip())
        elif '보합' in change_str:
            return 0.0
    try:
        return float(change_str)
    except:
        return 0.0

df['전일비'] = df['전일비'].apply(preprocess_change)

# 4. 슬라이싱 함수 정의 (5일 기준)
def create_sliding_dataset(df, window_size=5):
    samples = []
    for code, group in df.groupby('종목코드'):
        group = group.reset_index(drop=True)
        for i in range(len(group) - window_size):
            window = group.iloc[i:i+window_size]
            target_day = group.iloc[i+window_size]

            input_features = window[['종가', '전일비', '시가', '고가', '저가', '거래량']].astype(float).values.flatten()
            current_close = window.iloc[-1]['종가']
            next_close = target_day['종가']
            label = 1 if next_close > current_close else 0

            samples.append({
                '종목코드': code,
                '끝날짜': window.iloc[-1]['날짜'],
                **{f'X{i+1}': val for i, val in enumerate(input_features)},
                'Target': label
            })
    return pd.DataFrame(samples)

# 5. 학습용 슬라이싱 데이터 생성 (5일)
train_5 = create_sliding_dataset(df, window_size=5)

# 6. 저장
train_5.to_csv("Train_5days.csv", index=False, encoding='utf-8-sig')

# 7. 요약 출력 (선택사항)
print("총 샘플 수:", len(train_5))
print("종목 수:", train_5['종목코드'].nunique())
print(train_5.head())
