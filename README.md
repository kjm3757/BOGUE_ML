# BOGUE_ML – Campus Cafe Sales Forecasting

이 프로젝트는 **학사 일정 + POS 매출 데이터**를 기반으로  
캠퍼스 카페의 **일 매출을 예측하는 머신러닝 모델(LightGBM, XGBoost)** 구현 프로젝트입니다.

데이터 전처리 → Feature Engineering → 모델 학습 → Test 예측까지 하나의 파이프라인으로 구성되어 있습니다.


---

## 📁 Project Structure

BOGUE_ML/
├─ Data/
│ ├─ Feature.xlsx
│ ├─ POS_train_val.csv
│ └─ POS_test.csv
├─ LGBM.py
├─ XGB.py
├─ requirements.txt
└─ README.md

---

## 🔧 Environment Setup (환경 설정)

아래 명령어로 필요한 패키지를 한 번에 설치할 수 있습니다.

```bash
pip install -r requirements.txt

✔ requirements.txt 내용
pandas
numpy
scikit-learn
lightgbm
xgboost
openpyxl


📌 Data Description

Feature.xlsx
학사 일정 기반 Feature (요일, 학기/방학, 공휴일, 시험 일정 등)

POS_train_val.csv
2023~2024 POS 매출 데이터 (Train/Val)

POS_test.csv
예측 대상 Test 데이터

🛠 실행 방법

1. LightGBM 모델 실행
python LGBM.py

2. XGBoost 모델 실행
python XGB.py

🎯 주요 기능 요약
✔ Feature Engineering

Lag Features (1, 2, 3, 7, 14, 28)

Rolling Mean/Std (7, 14, 28)

시험 기간 window(exam_before3, exam_after3)

학기 × 주말 교차항

운영시간 Feature 자동 계산

요일 One-hot Encoding

✔ Model

LightGBM (LGBMRegressor)

XGBoost (XGBRegressor)

✔ Metrics

MAE

RMSE

SMAPE

📊 Output Example

Validation 성능 출력

Test 예측 성능 출력

Feature Importance (LightGBM)

Test 예측 결과 테이블 (상위 20개)

👥 Contributors

Team BOGUE ML

강민서 김정민 성세은