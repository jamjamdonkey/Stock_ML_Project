import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_top_kospi_50():
    headers = {'User-Agent': 'Mozilla/5.0'}
    stock_list = []

    for page in range(1, 2):  # 1~50위 = 페이지 1개만 사용
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

# 실행 및 종목코드 출력
if __name__ == "__main__":
    df_top50 = get_top_kospi_50()
    codes = df_top50['종목코드'].tolist()
    print('"' + '", "'.join(codes) + '"')
