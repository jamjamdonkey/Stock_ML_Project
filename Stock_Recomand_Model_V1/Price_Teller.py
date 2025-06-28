import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time

# 현재 파일 기준 절대 경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
recommand_path = os.path.join(BASE_DIR, "Top5_Recommanded.csv")
output_path = os.path.join(BASE_DIR, "Top5_Price_Indicators_Comment.csv")

# 보조지표 계산 함수
def calc_indicators(df):
    df = df.sort_values('날짜').copy()
    df['MA_20'] = df['종가'].rolling(20).mean()
    df['EMA_20'] = df['종가'].ewm(span=20, adjust=False).mean()
    df['BB_MID'] = df['종가'].rolling(20).mean()
    df['BB_STD'] = df['종가'].rolling(20).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    df['TR'] = df[['고가', '저가']].apply(lambda x: x[0] - x[1], axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    delta = df['종가'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['VOL_MA_5'] = df['거래량'].rolling(window=5).mean()
    return df

# 진단 코멘트 함수
def generate_comment(df):
    row = df.iloc[-1]
    comments = []
    rsi = row['RSI_14']
    close = row['종가']
    ema = row['EMA_20']
    ma = row['MA_20']
    upper = row['BB_UPPER']
    lower = row['BB_LOWER']
    atr = row['ATR_14']
    atr_mean = df['ATR_14'].mean()
    vol = row['거래량']
    vol_ma = row['VOL_MA_5']

    if rsi > 70:
        comments.append(f"RSI({rsi:.1f})는 과열 구간입니다.")
    elif rsi < 30:
        comments.append(f"RSI({rsi:.1f})는 과매도 상태입니다.")
    else:
        comments.append(f"RSI({rsi:.1f})는 중립 상태입니다.")

    if close >= upper:
        comments.append("볼린저밴드 상단 돌파: 단기 과열 가능성.")
    elif close <= lower:
        comments.append("볼린저밴드 하단 접근: 반등 가능성.")

    if close > ema and close > ma:
        comments.append("현재 추세는 상승 중입니다.")
    elif close < ema and close < ma:
        comments.append("현재 추세는 하락 중입니다.")
    else:
        comments.append("추세는 혼조 상태입니다.")

    if vol > 1.5 * vol_ma:
        comments.append("거래량이 평균보다 높습니다. 수급 변화 주의.")

    if atr > 1.3 * atr_mean:
        comments.append("최근 변동성이 확대되고 있습니다. 단기 리스크 유의.")

    return " ".join(comments)

# 크롤링 함수
def get_recent_prices(code, days=60):
    base_url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df_total = pd.DataFrame()
    page = 1
    MAX_PAGE = 20

    while len(df_total) < days:
        if page > MAX_PAGE:
            print(f"{code}: 페이지 {MAX_PAGE} 초과. 중단")
            break

        url = f"{base_url}&page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.find("table", class_="type2")
            tables = pd.read_html(StringIO(str(table)), match="날짜")
            if not tables:
                print(f"{code}: 테이블 파싱 실패")
                break
            df = tables[0]
            if df.shape[1] < 6:
                print(f"{code}: 칼럼 수 부족")
                break
            df = df.dropna(subset=["날짜", "종가", "시가", "고가", "저가", "거래량"])
        except Exception as e:
            print(f"{code}: 크롤링 실패: {e}")
            break

        df.columns = ["날짜", "종가", "전일비", "시가", "고가", "저가", "거래량"]
        df["종목코드"] = code
        df_total = pd.concat([df_total, df], ignore_index=True)
        page += 1
        time.sleep(0.3)

    try:
        df_total["날짜"] = pd.to_datetime(df_total["날짜"])
        df_total = df_total.sort_values("날짜").reset_index(drop=True)
    except:
        return pd.DataFrame()

    if len(df_total) < 40:
        print(f"{code}: 수집된 데이터 부족. 건너뜀.")
        return pd.DataFrame()

    return df_total.tail(days).reset_index(drop=True)

# 종목코드 로딩
try:
    top_df = pd.read_csv(recommand_path, encoding='utf-8-sig')
    top_codes = top_df['종목코드'].astype(str).apply(lambda x: x.zfill(6)).tolist()
except Exception as e:
    print(f"Top5_Recommanded.csv 로딩 실패: {e}")
    exit()

# 실행
result_rows = []

for code in top_codes:
    print(f"{code} 처리 중...")
    df = get_recent_prices(code, days=60)
    if df.empty:
        print(f"{code}: 시세 없음 또는 데이터 부족. 건너뜀.")
        continue

    df = calc_indicators(df)
    latest = df.iloc[-1]
    comment = generate_comment(df)

    row = {
        '종목코드': code,
        '날짜': latest['날짜'].date(),
        '종가': latest['종가'],
        'RSI_14': round(latest['RSI_14'], 2),
        'ATR_14': round(latest['ATR_14'], 2),
        'MA_20': round(latest['MA_20'], 2),
        'EMA_20': round(latest['EMA_20'], 2),
        'BB_UPPER': round(latest['BB_UPPER'], 2),
        'BB_LOWER': round(latest['BB_LOWER'], 2),
        'VOL_MA_5': int(latest['VOL_MA_5']),
        '진단': comment
    }

    result_rows.append(row)

# 최종 CSV 저장
if result_rows:
    final_df = pd.DataFrame(result_rows)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Top5_Price_Indicators_Comment.csv 저장 완료")

    # 전체 요약 출력
    print("전체 진단 결과 요약:")
    for row in result_rows:
        print(f"{row['종목코드']} | {row['날짜']} | 종가: {row['종가']}")
        print(f"진단: {row['진단']}")
        print("-" * 80)
else:
    print("저장할 결과가 없습니다.")
