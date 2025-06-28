import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import datetime
import os

# 네이버 증권 API 크롤러 클래스
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


# 시가총액 상위 50개 종목 코드 가져오기
def get_top_kospi_50():
    stock_list = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = 'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1'
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


# 🎯 실행: 원하는 날짜 범위 자동 반복 + 휴장일/중복 저장 방지 기능 포함
if __name__ == "__main__":
    from datetime import datetime, timedelta
    import os
    import pandas as pd

    # 🔽 분석 대상 날짜 범위 설정
    start_date = datetime.strptime("20250102", "%Y%m%d")
    end_date   = datetime.strptime("20250110", "%Y%m%d")
    cur_date   = start_date

    while cur_date <= end_date:
        target_date = cur_date.strftime("%Y%m%d")
        prev_date   = (cur_date - timedelta(days=1)).strftime("%Y%m%d")
        print(f"\n📅 실행 중: {target_date}")

        try:
            # 1. 종목 코드 수집
            top50_df = get_top_kospi_50()
            stock_codes = top50_df["종목코드"].tolist()

            # 2. 주가 정보 수집
            api = NaverStockAPI()
            df = api.get_today_multiple_stocks(stock_codes)

            if df.empty:
                print(f"❌ {target_date}: 데이터 없음 (주말 or 휴장일 가능)")
                cur_date += timedelta(days=1)
                continue

            # 3. 선택 피처 추출
            selected_columns = [
                "종목코드", "종목명", "날짜",
                "종가", "52주최고", "52주최저",
                "정적_거래량", "동적_거래량",
                "시가총액_원", "PER_숫자", "EPS_숫자",
                "전일대비", "개인순매수", "기관순매수", "외국인순매수"
            ]
            eda_df = df[selected_columns]

            # 4. 전날 삼성전자 지표와 비교
            eda_path = f"EDA_CSV/EDA_{target_date}.csv"
            prev_path = f"EDA_CSV/EDA_{prev_date}.csv"
            skip_save = False

            if os.path.exists(prev_path):
                try:
                    prev_df = pd.read_csv(prev_path)

                    # 삼성전자 데이터 유무 확인
                    if "005930" in eda_df["종목코드"].values and "005930" in prev_df["종목코드"].values:
                        today_row = eda_df[eda_df["종목코드"] == "005930"].iloc[0]
                        prev_row  = prev_df[prev_df["종목코드"] == "005930"].iloc[0]
                        key_cols = ["종가", "PER_숫자", "EPS_숫자", "동적_거래량"]

                        if all(today_row[col] == prev_row[col] for col in key_cols):
                            print(f"⏩ {target_date}: 주요 지표 동일 → 저장 생략")
                            skip_save = True
                    else:
                        print(f"⚠️ 삼성전자 데이터 없음 → 저장 진행")
                except Exception as e:
                    print(f"⚠️ 삼성전자 비교 중 예외 발생 → 저장 진행 (에러: {e})")

            # 5. 저장 (CSV + EDA)
            if not skip_save:
                os.makedirs("CSV", exist_ok=True)
                csv_file_name = f"{target_date}.csv"
                df.to_csv(os.path.join("CSV", csv_file_name), index=False, encoding="utf-8-sig")
                print(f"✔ 저장 완료: CSV/{csv_file_name}")

                os.makedirs("EDA_CSV", exist_ok=True)
                eda_df.to_csv(eda_path, index=False, encoding="utf-8-sig")
                print(f"📊 [EDA 저장 완료] {eda_path}")

        except Exception as e:
            print(f"❗ 오류 발생: {target_date} → {e}")

        # 날짜 1일 증가
        cur_date += timedelta(days=1)
