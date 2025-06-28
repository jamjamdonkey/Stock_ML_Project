import pandas as pd

# 1. 데이터 불러오기
df = pd.read_csv("Supervised_Learning_CSV/3Y_Merged_Stock_Data.csv")
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(by=['종목코드', '날짜'])

# 2. 전일비 전처리 함수 (같이 써야 함)
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

# 3. 슬라이싱 함수 (7일 버전)
def create_sliding_dataset(df, window_size=7):
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

# 4. 생성 및 저장
train_7 = create_sliding_dataset(df, window_size=7)
train_7.to_csv("Train_7days.csv", index=False, encoding='utf-8-sig')

# 5. 확인 (선택)
print("총 샘플 수:", len(train_7))
print("종목 수:", train_7['종목코드'].nunique())
print(train_7.head())
