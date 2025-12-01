# 📊 BOGUE_ML – Campus Cafe Sales Forecasting

이 프로젝트는 **학사 일정 데이터 + POS 매출 데이터**를 이용해  
캠퍼스 카페의 **일일 매출을 예측하는 머신러닝 모델**을 구현합니다.

데이터 전처리 → 피처 엔지니어링 → 모델 학습 → 테스트 예측까지  
완전한 머신러닝 파이프라인으로 구성되어 있습니다.

---

## 📁 Project Structure

```

BOGUE_ML/
├─ Data/
│   ├─ Feature.xlsx
│   ├─ POS_train_val.csv
│   └─ POS_test.csv
├─ description/
├─ final_code/
│   ├─ LGBM_tuning.py
│   ├─ XGB_tuning.py
│   └─ GRU_final.py
│   └─ LSTM_final.py
├─ ipynb/
├─ result/
├─ test_code/
├─ LICENSE
├─README.md
└─ requirements.txt

````

---

## 🛠️ Environment Setup

아래 명령어 한 번으로 환경을 세팅할 수 있습니다.

```bash
pip install -r requirements.txt
````

### ✔ requirements.txt 내용

```
pandas
numpy
scikit-learn
lightgbm
xgboost
openpyxl
```

---

## 📦 Data Description

**Feature.xlsx**

* 날짜별 학사 일정 Feature
* 요일, 방학/학기 여부
* 공휴일, 시험 일정 등 포함

**POS_train_val.csv**

* POS 매출 데이터 (Train/Validation)

**POS_test.csv**

* 예측 대상 Test 데이터

---

## 🚀 How to Run

### ▶ LightGBM 모델 실행

```bash
python LGBM.py
```

### ▶ XGBoost 모델 실행

```bash
python XGB.py
```

### ▶ GRU 모델 실행

```bash
python GRU_final.py
```

### ▶ LSTM 모델 실행

```bash
python LSTM_final.py
```

---

## 🧠 Feature Engineering Overview

본 프로젝트에서는 20개 이상의 Feature가 자동 생성됩니다.

### ✔ Time-Series Features

* Lag Features: `Lag1`, `Lag2`, `Lag3`, `Lag7`, `Lag14`, `Lag28`
* Rolling Means: `RollingMean7`, `RollingMean14`, `RollingMean28`
* Rolling Stds: `RollingStd7`, `RollingStd14`, `RollingStd28`

### ✔ Academic Calendar Features

* 학기/방학 구분
* 시험 기간 window: `exam_before3`, `exam_after3`
* 주말 여부 `weekend`
* 학기 × 주말 교차항 `semester_weekend`

### ✔ Custom Operating Hours

* 요일 + 학기 + 공휴일 기반 카페 운영시간 자동 계산
  (예: 월~금 12시간, 토 7시간, 일요일 0시간 등)

### ✔ Categorical Features

* 요일(weekday) → One-hot encoding

---

## 📈 Model Overview

### 🔹 LightGBM

* `Light Gradient Boosting Model`
* 빠르고 효율적인 트리 기반 모델
* Feature importance 확인 가능

### 🔹 XGBoost

* `eXtreme Gradient Boosting Model`
* 강력한 성능의 boosting 모델
* 자동 overfitting 방지 기능 포함

### 🔹 GRU
* 'Gated Recurrent Unit Model'
* 가벼운 구조의 시계열 딥러닝 모델
* 적은 파라미터로 빠르게 학습하고 장기 의존성도 처리 가능

### 🔹 LSTM
* 'Long Short-Term Memory Model'
* 복잡한 패턴을 잘 잡아내는 고성능 시계열 딥러닝 모델
* 장기 의존성 문제를 효과적으로 해결해 안정적인 예측 가능

---

## 🎯 Evaluation Metrics

모델 성능은 다음 3개 지표로 평가합니다.

* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Squared Error)
* **SMAPE** (Symmetric Mean Absolute Percentage Error)

---

## 📊 Output Example

* Validation 성능 출력
* Test 성능 출력
* LightGBM Feature Importance (상위 30개)
* Test 예측 결과 테이블 (상위 20개)

---

## 👥 Contributors

**Team BOGUE**

* 강민서 김정민 성세은
