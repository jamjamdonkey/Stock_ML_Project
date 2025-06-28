import os
import pandas as pd
from datetime import datetime

# 오늘 날짜
today = datetime.today().strftime('%Y%m%d')

# 경로 설정
input_path = f"C:/Users/JAMJAM/Stock_Project/Stock_Project/CSV/네이버api_{today}.csv"
output_folder = r"C:/Users/JAMJAM/Stock_Project/Stock_Project/EDA_CSV"
os.makedirs(output_folder, exist_ok=True)

# ✅ 수정된 파일명
file_name = f"EDA_{today}.csv"
output_path = os.path.join(output_folder, file_name)

# 피쳐 선택
selected_columns = [
    "종목코드", "종목명", "날짜",
    "시가총액_원", "PER_숫자", "EPS_숫자",
    "전일대비", "개인순매수", "기관순매수", "외국인순매수"
]

# CSV 불러오기 및 저장
df = pd.read_csv(input_path)
df[selected_columns].to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"[저장 완료] {output_path}")

# EDA_SET.py
import os
import pandas as pd
from datetime import datetime

# 오늘 날짜
today = datetime.today().strftime('%Y%m%d')

# 경로 설정
input_path = f"C:/Users/JAMJAM/Stock_Project/Stock_Project/CSV/네이버api_{today}.csv"
output_folder = r"C:/Users/JAMJAM/Stock_Project/Stock_Project/EDA_CSV"
os.makedirs(output_folder, exist_ok=True)

# ✅ 수정된 파일명
file_name = f"EDA_{today}.csv"
output_path = os.path.join(output_folder, file_name)

# 피쳐 선택
selected_columns = [
    "종목코드", "종목명", "날짜",
    "시가총액_원", "PER_숫자", "EPS_숫자",
    "전일대비", "개인순매수", "기관순매수", "외국인순매수"
]

# CSV 불러오기 및 저장
df = pd.read_csv(input_path)
df[selected_columns].to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"[저장 완료] {output_path}")

