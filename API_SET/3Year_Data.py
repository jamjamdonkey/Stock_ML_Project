import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# 📌 TOP 50 종목코드
stock_codes = [
    "005930", "000660", "207940", "373220", "035420", "034020", "105560", "012450", "005380", "329180",
    "005935", "000270", "068270", "035720", "055550", "009540", "028260", "042660", "012330", "032830",
    "402340", "086790", "011200", "015760", "064350", "005490", "138040", "000810", "323410", "267260",
    "259960", "010130", "316140", "096770", "033780", "010140", "018260", "051910", "034730", "024110",
    "006400", "003550", "030200", "352820", "006800", "066570", "377300", "017670", "079550", "272210"
]

# 📌 네이버 일별 시세 크롤러
def get_naver_day_price(code, max_pages=80):
    base_url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df_total = pd.DataFrame()

    for page in range(1, max_pages + 1):
        url = f"{base_url}&page={page}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="type2")
        df = pd.read_html(str(table))[0]
        df = df.dropna()
        df.columns = ["날짜", "종가", "전일비", "시가", "고가", "저가", "거래량"]
        df["종목코드"] = code
        df_total = pd.concat([df_total, df], ignore_index=True)
        time.sleep(0.5)

    df_total["날짜"] = pd.to_datetime(df_total["날짜"])

    # 📌 날짜를 가장 왼쪽으로 재정렬
    cols = ["날짜"] + [col for col in df_total.columns if col != "날짜"]
    df_total = df_total[cols]

    return df_total

# 📌 날짜별로 파일 저장
def save_by_date(df, save_folder="CSV_By_Date"):
    os.makedirs(save_folder, exist_ok=True)
    grouped = df.groupby("날짜")
    for date, group in grouped:
        date_str = date.strftime("%Y%m%d")
        # 날짜가 이미 컬럼으로 포함되어 있으므로 순서만 유지
        group = group[["날짜", "종목코드", "종가", "전일비", "시가", "고가", "저가", "거래량"]]
        group.to_csv(f"{save_folder}/{date_str}.csv", index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {date_str}.csv")

# 📌 전체 실행
all_data = []
for code in stock_codes:
    print(f"📥 수집 중: {code}")
    try:
        df = get_naver_day_price(code, max_pages=80)  # 약 3년치
        all_data.append(df)
    except Exception as e:
        print(f"❗ 실패: {code}, 이유: {e}")

merged_df = pd.concat(all_data, ignore_index=True)
save_by_date(merged_df)
