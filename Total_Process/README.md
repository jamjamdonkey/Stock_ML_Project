# 🔄 Total_Process Workflow

> 이 폴더는 전체 주식 예측 프로젝트의 주요 처리 파이프라인을 구성하며, 프로토타입 구동 과정 시각화를 목적으로 합니다. 데이터 전처리부터 클러스터링 및 SHAP 기반 피처 중요도 분석까지의 과정을 포함합니다.

---

## 📁 포함 파일 목록

| 파일명 | 설명 |
|--------|------|
| `3Y7D_LGBM_XGB_LR_SVM_Test.py` | 앙상블 모델 최종 백테스트 스크립트 |
| `3Y_CSV_7D_Slice.py` | 7일 슬라이싱 + 라벨 생성 |
| `3Y_Cluster_Combine.py` | 클러스터 ID 병합 |
| `3Y_Merged_Stock_Data.csv` | 통합된 3년치 종목별 시세 데이터 |
| `ByCluster_Analysis.py` | 클러스터별 수익률, 샘플 수, 통계 요약 |
| `CSV_COM.py` | CSV 병합 처리 코드 |
| `Clustering.py` | KMeans 클러스터링 및 시각화 생성 |
| `Model_Rolling_Results_WithCluster.csv` | 전체 백테스트 결과 기록 |
| `README.md` | 본 문서 |
| `SHAP_Analysis_LGBM_XGB.py` | SHAP 분석용 학습 코드 |
| `SHAP_LGBM_Korean.png` | LGBM 모델의 SHAP summary plot |
| `SHAP_XGB_Korean.png` | XGB 모델의 SHAP summary plot |
| `Stock_Clusters.csv` | 클러스터 ID가 붙은 종목 리스트 |
| `Train_7days_K4_Upgrade_3Y.csv` | 클러스터 없이 생성된 학습셋 |
| `Train_7days_K4_Upgrade_3Y_WithCluster.csv` | 클러스터 포함 최종 학습셋 |

---

## 🧭 전체 처리 순서

1. **CSV 데이터 병합 및 통합**  
   └─ `3Y_Merged_Stock_Data.csv`

2. **슬라이싱 및 클러스터링**  
   ├─ `3Y_CSV_7D_Slice.py`  
   ├─ `Clustering.py` (KMeans 기반)  
   └─ `Stock_Clusters.csv` / `output.png`

3. **학습셋 구성**  
   └─ `Train_7days_K4_Upgrade_3Y_WithCluster.csv`

4. **SHAP 분석**  
   ├─ `SHAP_Analysis_LGBM_XGB.py`  
   ├─ `SHAP_LGBM_Korean.png`  
   └─ `SHAP_XGB_Korean.png`

5. **클러스터별 성능 분석**  
   └─ `ByCluster_Analysis.py`

6. **백테스트 수행**  
   └─ `3Y7D_LGBM_XGB_LR_SVM_Test.py`

---

## ✅ 핵심 결과 요약

- 클러스터 3번의 **3일 평균 수익률** 및 **10% 이상 상승 비율**이 가장 높음
- 주요 피처 영향력: `range_pct`, `tail_up`, `클러스터`, `volume_price`
- 앙상블 모델 (XGB+LGBM+LR+SVM) 기준 Top-1 추천 시 수익률 평균 5% 이상

---

## 🧠 향후 계획

- 실시간 데이터 기반 학습 → 매일 아침 Top-N 종목 추천
- 키움증권 모의투자 대회에 실제 적용 및 리밸런싱 알고리즘 추가
- 예측 + 리스크 필터링 + 운용 로직 통합한 퀀트 시스템 구축

---


