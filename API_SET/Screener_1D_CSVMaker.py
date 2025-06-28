import pandas as pd
import numpy as np

# 1. CSV 불러오기
df = pd.read_csv("Supervised_Learning_CSV/Train_7days_K4_Upgrade.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('\ufeff', '', regex=True)  # BOM 제거
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values(['종목코드', '날짜'])

# 2. 파생 피처 생성
df['range_pct'] = (df['고가'] - df['저가']) / df['시가']
df['close_vs_open'] = (df['종가'] - df['시가']) / df['시가']
df['tail_up'] = (df['고가'] - df[['종가', '시가']].max(axis=1)) / df['시가']
df['tail_down'] = (df[['종가', '시가']].min(axis=1) - df['저가']) / df['시가']
df['body_ratio'] = abs(df['종가'] - df['시가']) / (df['고가'] - df['저가'])
df['volume_price'] = df['거래량'] * df['종가']
df['volatility'] = (df['고가'] - df['저가']) / df['종가']
df['mean_close_5'] = df.groupby('종목코드')['종가'].rolling(5).mean().reset_index(0, drop=True)
df['std_close_5'] = df.groupby('종목코드')['종가'].rolling(5).std().reset_index(0, drop=True)
df['close_vs_ma5'] = (df['종가'] - df['mean_close_5']) / df['mean_close_5']

# 3. 미래 수익률 (2일 후)
df['future_return_2d'] = df.groupby('종목코드')['종가'].pct_change(periods=2).shift(-2)

# 4. 타겟 라벨 생성 (예: 2일 후 수익률 > 1%면 1, 아니면 0)
df['target'] = df['future_return_2d'].apply(lambda x: 1 if x > 0.01 else 0)

# 5. NaN 제거 (학습에 필요한 열만 남기기)
required_cols = [
    'range_pct', 'close_vs_open', 'tail_up', 'tail_down', 'body_ratio',
    'volume_price', 'volatility', 'mean_close_5', 'std_close_5',
    'close_vs_ma5', 'future_return_2d', 'target'
]
df_final = df.dropna(subset=required_cols)

# 6. CSV 저장
df_final.to_csv("Stock_Screener_FeatureSet_2D.csv", index=False, encoding='utf-8-sig')
print("✅ 저장 완료: Stock_Screener_FeatureSet_2D.csv")

