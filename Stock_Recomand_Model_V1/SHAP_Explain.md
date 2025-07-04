---

### 1. `close_1`
- **설명**: 전일 종가 (하루 전)
- **자연어 해석**: 어제의 종가입니다. 주가의 직전 흐름을 반영합니다.
- **투자 관점**: 최근 주가 흐름을 가장 직접적으로 나타내며, 단기 모멘텀 판단에 중요합니다.

### 2. `close_2`
- **설명**: 이틀 전 종가
- **자연어 해석**: 이틀 전 주식 마감 가격입니다.
- **투자 관점**: 2일 연속 상승/하락 여부를 통해 추세 판단이 가능합니다.

### 3. `close_3`
- **설명**: 3일 전 종가
- **자연어 해석**: 3일 전 주식의 종가입니다.
- **투자 관점**: 단기 패턴(예: V자 반등, 횡보 등)의 기초가 되는 정보입니다.

### 4. `mean_close`
- **설명**: 최근 7일간 종가 평균
- **자연어 해석**: 최근 일주일 간 평균 종가입니다.
- **투자 관점**: 안정성 판단 지표로, 평균보다 현재 주가가 높으면 과열 가능성도 시사합니다.

### 5. `std_close`
- **설명**: 최근 7일간 종가 표준편차
- **자연어 해석**: 일주일 간 주가의 흔들림(변동성) 정도입니다.
- **투자 관점**: 높은 값은 불안정성을, 낮은 값은 안정적인 흐름을 나타냅니다.

### 6. `return_1d`
- **설명**: 1일 수익률
- **자연어 해석**: 어제 하루 동안의 주가 등락률입니다.
- **투자 관점**: 단기 상승 또는 급락 신호 탐지에 사용됩니다.

### 7. `return_3d`
- **설명**: 최근 3일간 수익률
- **자연어 해석**: 최근 3일간의 주가 수익률 합산입니다.
- **투자 관점**: 단기 모멘텀 강도 판단에 유용합니다.

### 8. `num_up_days`
- **설명**: 7일 중 상승한 일수
- **자연어 해석**: 최근 7일 동안 주가가 오른 날의 수입니다.
- **투자 관점**: 우상향 흐름 여부 파악에 직관적입니다.

### 9. `시가`
- **설명**: 당일 시가 (시장 시작 가격)
- **자연어 해석**: 오늘 장 시작 시의 가격입니다.
- **투자 관점**: 장 초반 투자심리 및 갭 상승 여부 파악에 유용합니다.

### 10. `고가`
- **설명**: 당일 고가
- **자연어 해석**: 오늘 장중 최고 가격입니다.
- **투자 관점**: 매수세 강도를 반영하며, 돌파 여부는 심리적 의미가 있습니다.

### 11. `저가`
- **설명**: 당일 저가
- **자연어 해석**: 오늘 장중 최저 가격입니다.
- **투자 관점**: 매도세 압박 또는 지지선 여부 판단에 쓰입니다.

### 12. `거래량`
- **설명**: 당일 거래량
- **자연어 해석**: 오늘 주식이 얼마나 많이 거래되었는지를 나타냅니다.
- **투자 관점**: 수급 강도 파악 및 이상 거래 감지에 유용합니다.

### 13. `range_pct`
- **설명**: (고가 - 저가) / 시가
- **자연어 해석**: 하루 동안 가격이 얼마나 출렁였는지 비율로 나타냅니다.
- **투자 관점**: 높은 변동성 종목 탐색, 데이트레이딩 참고 지표입니다.

### 14. `close_vs_open`
- **설명**: (종가 - 시가) / 시가
- **자연어 해석**: 장중 가격 흐름의 최종 결과입니다.
- **투자 관점**: 양봉(상승 마감) 또는 음봉(하락 마감) 여부 판단에 유용합니다.

### 15. `tail_up`
- **설명**: 고가와 종가/시가 중 큰 값의 차이 / 시가
- **자연어 해석**: 장중 윗꼬리 길이 비율입니다.
- **투자 관점**: 장중 매도 압박 여부, 위에서 눌림 여부 판단 지표입니다.

### 16. `tail_down`
- **설명**: 저가와 종가/시가 중 작은 값의 차이 / 시가
- **자연어 해석**: 장중 아랫꼬리 길이 비율입니다.
- **투자 관점**: 저점 매수세 유입 여부 판단에 유용합니다.

### 17. `volume_price`
- **설명**: 거래량 × 종가
- **자연어 해석**: 거래된 금액 총합입니다.
- **투자 관점**: 종목의 거래 활성도와 시장 관심도를 파악할 수 있습니다.

---

이 문서를 기반으로 SHAP 해석 출력을 유저들이 자연어로 이해할 수 있으며, 각 지표가 의미하는 바를 투자 판단에 연결할 수 있습니다. SHAP 분석의 투명성과 신뢰도를 동시에 높이는 기반 자료로 활용 가능합니다.

