# 📋 최종 앙상블 코드 상세 설명

## 🔍 데이터 누수 검토 결과

### ✅ **결론: 데이터 누수 없음 확인**

모든 모델에서 train/test 데이터가 완전히 분리되어 있으며, test 데이터의 실제 매출값은 예측 과정에서 전혀 사용되지 않습니다.

---

## 📚 전체 코드 구조 상세 설명

### **PART 0: 초기 설정 (라인 16-114)**

#### **1. Seed 고정 (재현성 보장)**
```python
SEED = 42
# Python, NumPy, PyTorch, CUDA 모든 seed 고정
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
```

**목적**: 같은 결과를 재현하기 위해 모든 랜덤 요소를 고정

#### **2. 데이터 로드 및 전처리**
```python
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
acad = pd.read_csv(ACAD_PATH)  # 학사일정 데이터
```

**학사일정 Merge**:
- Train과 Test 모두에 학사일정 정보 추가
- ✅ **누수 아님**: 학사일정은 미리 알 수 있는 정보

**일매출 정리**:
- 쉼표 제거, 숫자 변환
- Test 데이터의 일매출은 평가용으로만 사용

---

### **PART 1: LSTM 모델 (라인 116-398)**

#### **1.1 Feature Engineering**

**`make_basic_features_lstm(df)`** - 시간 정보만:
- `DayOfWeek`: 요일 (0=월요일, 6=일요일)
- `Month`, `Day`: 월, 일
- `IsWeekend`: 주말 여부
- **`OpHours`**: 영업 시간 (시간 단위)
  - 학기 중: 월-금 11시간, 토요일 6시간, 일요일 0시간
  - 방학 중: 월-토 6시간, 일요일 0시간
- `OpHoursFactor`: 0~1로 정규화 (11시간 기준)

**`make_features_lstm(df)`** - 전체 피처:
- 기본 피처 + 영업시간 피처
- **Lag 피처**: 1, 2, 3, 7, 14, 28일 전 매출
- **Rolling 피처**: 
  - `Mean7/14/28`: 7/14/28일 이동평균
  - `Std7/14/28`: 7/14/28일 표준편차
- `IsZeroSales`: 휴무일 여부 (binary)

**데이터 누수 검토 ✓**:
- ✅ Train 전체에서 feature 생성 후 split하는 것은 시계열 표준
- ✅ Val의 lag/rolling은 train 데이터를 참조하므로 문제 없음

#### **1.2 데이터 준비**

**Train/Val Split**:
```python
train_df_lstm, val_df_lstm = train_test_split(train_lstm, test_size=0.2, shuffle=False, random_state=SEED)
```
- `shuffle=False`: 시계열이므로 시간 순서 유지

**Feature Selection**:
- `meta_cols_lstm`: Lag/Mean/Std 제외한 모든 numeric 피처
- `tree_features_lstm`: meta_cols + Lag/Mean/Std 피처

**Scaling**:
- `MinMaxScaler`: 0~1 범위로 정규화
- Sales와 Meta 피처 각각 별도 스케일러 사용

#### **1.3 모델 구조**

**MetaLSTM**:
```
Input: (batch, lookback=21, input_dim)
  ↓
LSTM (2 layers, hidden=128, dropout=0.4)
  ↓
FC1 (128 → 64) + ReLU + Dropout
  ↓
FC2 (64 → 1)
  ↓
Output: (batch, 1)
```

**하이퍼파라미터**:
- Lookback: 21일
- Hidden size: 128
- Layers: 2
- Dropout: 0.4
- Learning rate: 0.0005
- Batch size: 16

#### **1.4 학습**

**Training Loop**:
1. Forward pass
2. Loss 계산 (MSE)
3. Backward pass
4. Gradient clipping (max_norm=1.0)
5. Optimizer step

**Early Stopping**:
- Patience: 15 epochs
- Min delta: 0.0001
- Best model 저장 및 복원

**Learning Rate Scheduling**:
- ReduceLROnPlateau: 검증 손실이 개선되지 않으면 LR 절반으로 감소

#### **1.5 예측**

**Autoregressive 예측**:
```python
predict_nn_autoreg_lstm(model, train_df, future_meta_df, lookback)
```

**과정**:
1. Train 데이터의 최근 21일을 history로 사용
2. 다음 날 예측
3. 예측값을 history에 추가
4. 다음 날 예측 시 이전 예측값 포함하여 사용
5. 반복

**데이터 누수 검토 ✓**:
- ✅ `future_df_lstm`에는 test의 일매출 없음 (날짜 + 학사일정만)
- ✅ History는 train 데이터만 사용
- ✅ 예측값만 다음 step에 사용

**Post-processing**:
```python
postprocess_zero_days_lstm(preds, future_dates, train_df, threshold_ratio=0.7, small_pred_threshold=10000)
```

**규칙**:
- Train 데이터에서 월-일별로 0인 비율 계산
- 70% 이상 0인 날짜 + 예측값 < 10,000 → 0으로 강제

---

### **PART 2: GRU 모델 (라인 400-765)**

#### **2.1 Feature Engineering**

**`make_features_gru(df)`**:
- LSTM과 유사 + 추가 피처:
  - `Mean3`: 3일 이동평균
  - `Max7`, `Min7`: 7일 최대/최소값
  - `CV7`: 변동계수 (Std7 / Mean7)
  - `MonthAvg`: 월별 평균 매출
  - `WeekdayAvg`: 요일별 평균 매출

**`make_features_gru_safe(df, month_avg_train, weekday_avg_train)`**:
- ✅ **데이터 누수 방지 버전**
- Train의 평균값을 파라미터로 받아 사용
- Test 데이터에서 직접 평균 계산 안 함

#### **2.2 모델 구조**

**MetaGRU**:
```
Input: (batch, lookback=7, input_dim)
  ↓
GRU (2 layers, hidden=64, dropout=0.3)
  ↓
FC1 (64 → 32) + ReLU + Dropout
  ↓
FC2 (32 → 1)
  ↓
Output: (batch, 1)
```

**하이퍼파라미터**:
- Lookback: 7일 (LSTM보다 짧음)
- Hidden size: 64
- Layers: 2
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 64

#### **2.3 예측 (Direct Multi-step)**

**`predict_gru_direct_safe`**:
- ✅ Train의 평균값 사전 계산:
  ```python
  month_avg_train_gru = train_full_gru_for_avg.groupby("Month")["일매출"].mean()
  weekday_avg_train_gru = train_full_gru_for_avg.groupby("DayOfWeek")["일매출"].mean()
  ```

**예측 과정**:
1. Train 데이터를 history로 초기화
2. 각 step마다:
   - 필요하면 이전 예측값으로 sequence 구성
   - **Train의 평균값만 사용**하여 MonthAvg, WeekdayAvg 생성
   - 7일 sequence로 예측
   - 예측값을 history에 추가
3. 반복

**데이터 누수 검토 ✓**:
- ✅ `future_df_gru`는 날짜 + 학사일정만 (일매출 없음)
- ✅ Train의 평균값만 사용 (test 데이터 사용 안 함)
- ✅ History는 train + 예측값만 사용

---

### **PART 3: Tree 모델 (라인 767-1006)**

#### **3.1 Feature Engineering**

**`make_features_tree(df)`**:
- GRU와 동일한 피처 생성
- MonthAvg, WeekdayAvg 포함

#### **3.2 모델 학습**

**LightGBM**:
- Objective: regression (RMSE)
- num_leaves: 100
- learning_rate: 0.01
- max_depth: 10
- Regularization: alpha=0.1, lambda=2.0

**XGBoost**:
- Objective: reg:squarederror (RMSE)
- max_depth: 5
- learning_rate: 0.01
- Regularization: alpha=0.1, lambda=2.0

**전체 데이터 재학습**:
- Validation에서 찾은 best iteration으로 전체 train 데이터에 재학습
- ✅ Test 데이터 사용 안 함

#### **3.3 예측 (Autoregressive)**

**`predict_tree_autoreg`**:
- `history_df`로 `train_full_tree`만 전달
- 각 step마다:
  1. 현재까지의 history로 feature 생성
  2. 모델 예측
  3. **Scale 보정**:
     - 너무 작은 예측값 보정 (recent_7d_avg 기준)
     - 최소값 보장 (10,000)
  4. 예측값을 history에 추가하고 feature 재생성
  5. 반복

**데이터 누수 검토 ✓**:
- ✅ Test 데이터 전혀 사용 안 함
- ✅ `train_full_tree`만 사용
- ✅ 예측값으로만 다음 step 진행

---

### **PART 4: 최종 앙상블 (라인 1008-1035)**

#### **앙상블 구조**:

**1단계: NN 앙상블**
```python
future_nn_ensemble = (future_lstm + future_gru) / 2
```
- LSTM과 GRU의 단순 평균

**2단계: 1차 최종 앙상블**
```python
future_final = 0.3 * future_tree + 0.7 * future_nn_ensemble
```
- Tree 앙상블: 30%
- NN 앙상블: 70%

**3단계: 2차 최종 앙상블**
```python
future_final2 = 0.6 * future_nn_ensemble + 0.4 * future_final
```
- NN 앙상블: 60%
- 1차 최종 앙상블: 40%

**최종 가중치 분해**:
- Tree: 0.3 × 0.4 = 0.12 (12%)
- NN: 0.7 × 0.4 + 0.6 = 0.88 (88%)

---

### **PART 5: 성능 평가 및 저장 (라인 1037-1126)**

#### **Metrics**:
- **MAE** (Mean Absolute Error): 평균 절대 오차
- **RMSE** (Root Mean Squared Error): 평균 제곱근 오차
- **SMAPE** (Symmetric Mean Absolute Percentage Error): 대칭 평균 절대 백분율 오차

#### **결과 저장**:
- 모든 모델 예측값 + 최종 앙상블을 CSV로 저장
- `final_test_prediction_optimal.csv`

---

## 🔑 핵심 특징 요약

### **1. 데이터 누수 방지**
- ✅ Train/Test 완전 분리
- ✅ Train의 통계값만 사용 (MonthAvg, WeekdayAvg)
- ✅ Test 데이터의 실제 매출값 사용 안 함
- ✅ 예측값으로만 다음 step 진행

### **2. 영업시간 피처**
- 학기/방학 구분에 따른 실제 영업시간 반영
- 학기 중: 월-금 11시간, 토요일 6시간
- 방학 중: 월-토 6시간
- 일요일: 항상 0시간

### **3. 모델 다양성**
- **LSTM**: 21일 lookback, autoregressive
- **GRU**: 7일 lookback, direct multi-step
- **LightGBM/XGBoost**: Tree 기반, autoregressive

### **4. 다층 앙상블**
- 1단계: NN 앙상블 (LSTM + GRU)
- 2단계: 1차 최종 (Tree + NN)
- 3단계: 2차 최종 (NN + 1차 최종)

### **5. Post-processing**
- Zero-sales day 패턴 학습 및 적용
- Scale 보정 (Tree 모델)
- 최소값 보장

---

## ✅ 최종 검증

**데이터 누수 없음 확인**:
1. ✅ LSTM: Test 일매출 사용 안 함
2. ✅ GRU: Train 평균값만 사용
3. ✅ Tree: Test 데이터 전혀 사용 안 함
4. ✅ 학사일정: 미리 알 수 있는 정보 (누수 아님)

**코드 품질**:
- 재현성 보장 (Seed 고정)
- 명확한 구조 (PART 1/2/3 분리)
- 데이터 누수 방지 로직 명시

