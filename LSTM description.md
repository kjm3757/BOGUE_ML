<img width="843" height="273" alt="Frame 1" src="https://github.com/user-attachments/assets/87851a9d-d1b2-4bd9-bc91-7e09dcabed2b" />

초안

1. 모델 개요 (Model Overview)
모델명: LSTM 
구조: 3계층 LSTM 기반의 다대일(Many-to-One) 시퀀스 
목적: 테스트 기간의 어떤 실제 매출 값도 사용하지 않고, 오직 훈련된 모델과 예측된 값만을 사용하여 장기적인 매출 예측의 신뢰도를 평가합니다.
2. 핵심 방법론 (Core Methodology)
A. 특징 구성 (Feature Engineering)
시퀀스 입력
과거 28일간의 매출 이력 (LOOKBACK=28).
모델이 매출 패턴을 인식하는 데 사용.

Meta 특징
학사일정 등의 확실한 정보 사용 (요일, 월, 공휴일, 시험 기간 등).
데이터 누수 방지를 위해 매출 기반 특징(Lag/Rolling)은 모두 제외하고 순수 외부 정보만 활용.

데이터 처리
MinMax Scaling
매출과 Meta 특징을 0과 1 사이로 정규화하여 학습 안정성을 확보.

B. 엄격한 검증 방식: 순수 재귀적 예측 (Pure Recursive Forecasting)
이 모델은 실제 실전 예측 환경을 모방하여 오류 누적 효과를 측정합니다.
재귀적 예측 원리:
입력 갱신: 다음 날의 매출을 예측할 때, 이전 날의 실제 매출 값 대신, 모델이 직전에 예측한 값을 입력 시퀀스에 다시 넣어 예측 체인을 이어갑니다.
Meta 갱신: Meta 특징은 미리 알고 있는 달력 정보(df_test의 요일 등)를 사용하여 갱신합니다.
의의: 이 방식은 예측 오차가 시퀀스를 따라 누적되어 장기 예측의 신뢰도가 떨어지는 현상(평탄화)을 보여주므로, 모델의 내재적 예측 능력을 가장 엄격하게 평가합니다.

3. 최종 성능 평가 (Validation Metrics)
1. MAE: 112,531.24 KRW
2. RMSE: 162,578.40 KRW
3. SMAPE: 60.02 %
