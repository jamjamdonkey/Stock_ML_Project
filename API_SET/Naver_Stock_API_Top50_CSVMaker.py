#네이버 증권 API 클래스
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import datetime
import os

class NaverStockAPI:
    def __init__(self):
        self.base_url = "https://m.stock.naver.com/api/stock"

    def clean_market_cap(self, market_cap_str):
        if not market_cap_str:
            return None
        조_match = re.search(r"([\d,\.]+)조", market_cap_str)
        억_match = re.search(r"([\d,\.]+)억", market_cap_str)
        조 = float(조_match.group(1).replace(",", "")) if 조_match else 0
        억 = float(억_match.group(1).replace(",", "")) if 억_match else 0
        return int(조 * 10**12 + 억 * 10**8)

    def clean_number_with_units(self, num_str):
        if not num_str:
            return None
        cleaned = re.sub(r"[^\d\.]", "", num_str)
        try:
            return float(cleaned.replace(",", ""))
        except:
            return None

    def get_today_stock_info(self, stock_code):
        url = f"{self.base_url}/{stock_code}/integration"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            종목명 = data.get("stockName", "")
            total_infos = data.get("totalInfos", [])
            deal_trends = data.get("dealTrendInfos", [])
            def find_value(code):
                for info in total_infos:
                    if info.get("code") == code:
                        return info.get("value", "")
                return ""

            trend = deal_trends[0] if deal_trends else {}

            raw_market_cap = find_value("marketValue")
            raw_per = find_value("per")
            raw_eps = find_value("eps")

            return {
                "종목코드": stock_code,
                "종목명": 종목명,
                "날짜": trend.get("bizdate", ""),
                "시작가": find_value("openPrice"),
                "전일가": find_value("lastClosePrice"),
                "정적_거래량": find_value("accumulatedTradingVolume"),
                "시가총액": self.clean_market_cap(raw_market_cap),
                "PER": self.clean_number_with_units(raw_per),
                "EPS": self.clean_number_with_units(raw_eps),
                "52주최고": find_value("highPriceOf52Weeks"),
                "52주최저": find_value("lowPriceOf52Weeks"),
                "종가": trend.get("closePrice", ""),
                "전일대비": trend.get("compareToPreviousClosePrice", ""),
                "외국인순매수": trend.get("foreignerPureBuyQuant", ""),
                "기관순매수": trend.get("organPureBuyQuant", ""),
                "개인순매수": trend.get("individualPureBuyQuant", ""),
                "동적_거래량": trend.get("accumulatedTradingVolume", ""),
                "시가총액_원": self.clean_market_cap(raw_market_cap),
                "PER_숫자": self.clean_number_with_units(raw_per),
                "EPS_숫자": self.clean_number_with_units(raw_eps)
            }

        except requests.exceptions.RequestException as e:
            print(f"[에러] API 호출 실패: {e}")
            return None

    def get_today_multiple_stocks(self, stock_codes):
        results = []
        for code in stock_codes:
            info = self.get_today_stock_info(code)
            if info:
                results.append(info)
        return pd.DataFrame(results)


# -------------------- 시가총액 상위 50개 크롤러 --------------------
def get_top_kospi_50():
    stock_list = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for page in range(1, 2):  # 1페이지에 50위까지 나옴
        url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}'
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.select('table.type_2 tr')[2:]
        for row in table:
            link = row.select_one('a.tltle')
            if link:
                name = link.text.strip()
                href = link['href']
                code = href.split('=')[-1]
                stock_list.append({'종목명': name, '종목코드': code})
    return pd.DataFrame(stock_list).drop_duplicates().reset_index(drop=True)


# -------------------- 실행 스크립트 --------------------
if __name__ == "__main__":
    today = datetime.today().strftime('%Y%m%d')

    # 종목코드 수집
    top50_df = get_top_kospi_50()
    stock_codes = top50_df["종목코드"].tolist()

    # 주가정보 수집
    api = NaverStockAPI()
    df = api.get_today_multiple_stocks(stock_codes)

    # 저장
    csv_folder = "CSV"
    os.makedirs(csv_folder, exist_ok=True)
    file_name = f"{today}.csv"
    df.to_csv(os.path.join(csv_folder, file_name), index=False, encoding="utf-8-sig")

    print(f"✔ 저장 완료: {file_name}")

import os
import pandas as pd
from datetime import datetime

# 오늘 날짜
today = datetime.today().strftime('%Y%m%d')

# 경로 설정
input_path = f"C:/Users/JAMJAM/Stock_Project/Stock_Project/CSV/{today}.csv"
output_folder = r"C:/Users/JAMJAM/Stock_Project/Stock_Project/EDA_CSV"
os.makedirs(output_folder, exist_ok=True)

# ✅ 수정된 파일명
file_name = f"EDA_{today}.csv"
output_path = os.path.join(output_folder, file_name)

# 피쳐 선택
selected_columns = [
    "종목코드", "종목명", "날짜",
    "종가", "52주최고", "52주최저",
    "정적_거래량", "동적_거래량",
    "시가총액_원", "PER_숫자", "EPS_숫자",
    "전일대비", "개인순매수", "기관순매수", "외국인순매수"
]

# CSV 불러오기 및 저장
df = pd.read_csv(input_path)
df[selected_columns].to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"[저장 완료] {output_path}")
