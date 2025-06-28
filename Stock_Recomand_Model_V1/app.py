from flask import Flask, render_template, request
import os
import pandas as pd
from Auto_Upgrade_Crawling import run_crawling
from Recommand_Final_Tuning_Result import run_prediction
from Price_Teller import run_price_analysis

# Flask 앱 객체 생성
app = Flask(__name__, template_folder="templates")

# 현재 파일 기준 상위 경로
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECOMMAND_PATH = os.path.join(BASE_DIR, "Top5_Recommanded.csv")
COMMENT_PATH = os.path.join(BASE_DIR, "Top5_Price_Indicators_Comment.csv")

# SHAP 피처 설명 사전 (요약버전)
shap_explain = {
    "volume_price": "종가 × 거래량으로, 거래 에너지를 반영하는 지표입니다.",
    "num_up_days": "7일 중 상승 마감한 날의 수로, 상승 추세 여부를 반영합니다.",
    "std_close": "7일간 종가의 표준편차로, 주가의 변동성을 의미합니다.",
    "return_1d": "전일 대비 수익률입니다.",
    "return_3d": "3일 전 대비 수익률입니다.",
    "mean_close": "7일간 평균 종가로, 전체 가격 레벨을 나타냅니다.",
    "close_4": "4일 전 종가입니다.",
    "tail_up": "윗꼬리 비율로, 장중 고점에서 밀린 정도를 나타냅니다.",
    "tail_down": "아랫꼬리 비율로, 장중 저점 대비 반등 정도를 나타냅니다.",
    "close_vs_open": "시가 대비 종가의 위치입니다.",
    "range_pct": "고가-저가의 변동폭 비율입니다.",
    "종가": "해당일 종가입니다.",
    "저가": "해당일 저가입니다.",
    "고가": "해당일 고가입니다.",
    "시가": "해당일 시가입니다."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_model", methods=["POST"])
def run_model():
    try:
        # 1. 데이터 수집 및 분석 실행 (함수 기반)
        run_crawling()
        run_prediction()
        run_price_analysis()

        # 2. 결과 불러오기
        rec_df = pd.read_csv(RECOMMAND_PATH, encoding='utf-8-sig')
        comment_df = pd.read_csv(COMMENT_PATH, encoding='utf-8-sig')

        result_html = "<h3>✅ 추천 종목 결과</h3>"

        for _, row in rec_df.iterrows():
            code = str(row['종목코드']).zfill(6)
            prob = f"{row['예측확률']*100:.2f}%"
            result_html += f"<b>{code}</b> - 예측 확률: <b>{prob}</b><br>"
            for i in range(1, 4):
                feat = row[f"SHAP_Feature_{i}"]
                val = row[f"SHAP_Value_{i}"]
                impact = row[f"SHAP_Impact_{i}"]
                explain = shap_explain.get(feat, "설명 없음")
                sign = "+" if impact >= 0 else "-"
                result_html += f"&nbsp;&nbsp;&nbsp;▶ {feat} = {val:.4f} ({sign}{abs(impact):.4f}) → {explain}<br>"
            result_html += "<br>"

        result_html += "<h3>📈 기술적 분석 요약</h3>"

        for _, row in comment_df.iterrows():
            result_html += f"<b>{str(row['종목코드']).zfill(6)}</b> ({row['날짜']} 종가: {row['종가']})<br>"
            result_html += f"📝 {row['진단']}<br><br>"

        return render_template("index.html", result=result_html)

    except Exception as e:
        print(f"[ERROR] {e}")  # 내부 로깅
        return "<h3>❌ 예측 중 문제가 발생했습니다. 관리자에게 문의해주세요.</h3>"

if __name__ == "__main__":
    app.run(debug=False)
