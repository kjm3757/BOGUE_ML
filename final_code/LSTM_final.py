import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 0. 영업 시간 계산 유틸리티 함수 추가 (요일 기반 단순 가중치)
# --------------------------------------------------------------------------
def calculate_operating_hours(row):
    """
    요일(DayOfWeek)만을 기준으로 영업 시간 가중치를 계산합니다.
    0=월요일, 5=토요일, 6=일요일
    """
    weekday = row["DayOfWeek"]

    # 1. 일요일 미운영 (DayOfWeek = 6) -> 0.0
    if weekday == 6:
        return 0.0
    
    # 2. 토요일 (DayOfWeek = 5) -> 0.5
    if weekday == 5:
        return 0.5
    
    # 3. 평일 (월~금, DayOfWeek = 0~4) -> 1.0
    return 1.0


# --------------------------------------------------------------------------
# 1. 환경 설정 및 지표 정의
# --------------------------------------------------------------------------
set_seed = lambda x: np.random.seed(x) or torch.manual_seed(x)
set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 32, 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📌 CSV 파일 경로 정의
TRAIN_CSV = '../Data/POS_train_val.csv'
TEST_CSV = '../Data/POS_test.csv'
CALENDAR_CSV = '../Data/Feature.xlsx'

SALES_COL = 'daily'
DATE_COL = 'date'
GROUP_COL = '그룹키'

# Meta 특징 정의 (총 15개)
CALENDAR_BINARY_COLS = ['weekend', 'holiday', 'semester', 'seasonal', 'exam', 'ceremony'] 
DOW_COLS = [f'DOW_{i}' for i in range(7)]
OP_HOUR_COL = 'ScaledOperatingHours'
ALL_META_FEATURES = CALENDAR_BINARY_COLS + DOW_COLS + [OP_HOUR_COL]

INPUT_DIM = 1 + len(ALL_META_FEATURES) 

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 1e-6
    return np.mean(numerator / denominator) * 100

# --- 2. LSTM 모델 클래스 정의 (Meta 포함) ---
class SimpleLSTMWithMeta(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True) 
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        # 마지막 시점의 은닉 상태를 사용하여 예측 
        return self.fc(out[:, -1, :]) 

# --- 3. 데이터 로드 및 특징 생성 함수 (Sales + Meta 결합) ---
def create_data_for_lstm(train_csv, test_csv, calendar_csv):
    try:
        df_train_raw = pd.read_csv(train_csv)
        df_test_raw = pd.read_csv(test_csv)
        df_calendar_raw = pd.read_excel(calendar_csv)
    except Exception as e:
        print(f"Error: File loading failed. {e}"); return pd.DataFrame(), pd.DataFrame(), []

    # 1. POS 데이터 정리 함수
    def clean_pos_data(df):
        df.rename(columns={DATE_COL: 'date', SALES_COL: 'sales'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        def clean_sales(series):
            series = series.astype(str).str.replace(',', '', regex=False)
            return pd.to_numeric(series, errors='coerce').fillna(0)
        
        df['sales'] = clean_sales(df['sales'])
        df[GROUP_COL] = '전체'
        return df

    df_train = clean_pos_data(df_train_raw)
    df_test = clean_pos_data(df_test_raw)
    
    # 2. 학사일정 데이터 정리 및 Meta 특징 생성
    df_calendar = df_calendar_raw.copy()
    df_calendar.rename(columns={'date': 'date'}, inplace=True)
    df_calendar['date'] = pd.to_datetime(df_calendar['date'], errors='coerce')
    
    # DayOfWeek 특징 추가 (0=월요일, 6=일요일)
    df_calendar['DayOfWeek'] = df_calendar['date'].dt.dayofweek
    
    # 영업 시간 계산 및 추가
    df_calendar['OperatingHours'] = df_calendar.apply(calculate_operating_hours, axis=1)
    df_calendar[OP_HOUR_COL] = df_calendar['OperatingHours'] 
    
    # One-Hot Encoding: DayOfWeek (7개 특징)
    df_dow = pd.get_dummies(df_calendar['DayOfWeek'], prefix='DOW', dtype=float)
    df_calendar = pd.concat([df_calendar, df_dow], axis=1)
    
    # 최종 Meta Features 정의 (15개)
    meta_features = ALL_META_FEATURES
    
    # 3. 데이터 병합 (POS 데이터 + Meta 데이터)
    # 병합할 Meta 컬럼 목록 (date + 15개 특징)
    merge_cols = ['date'] + meta_features

    # 3. POS 데이터 + Meta 데이터 병합 (date는 건드리지 않고, meta만 0으로 채우기)
    df_train = pd.merge(df_train, df_calendar[merge_cols], on='date', how='left')
    df_test  = pd.merge(df_test,  df_calendar[merge_cols], on='date', how='left')

    # ❗ date 컬럼은 절대 fillna(0) 하지 않기
    # meta 특징들만 결측치 0으로 채우기
    df_train[meta_features] = df_train[meta_features].fillna(0.0)
    df_test[meta_features]  = df_test[meta_features].fillna(0.0)

    # 4. 최종 통합 데이터 정렬 및 인덱스 초기화
    df_train = df_train.sort_values('date').reset_index(drop=True)
    df_test  = df_test.sort_values('date').reset_index(drop=True)

    return df_train, df_test, meta_features


# --- 4. 훈련, 예측 및 검증 함수 ---
def train_predict_validate():
    df_train, df_test, meta_cols = create_data_for_lstm(TRAIN_CSV, TEST_CSV, CALENDAR_CSV)
    if df_train.empty or df_test.empty: return

    print(f"Train Data Period: {df_train['date'].min().date()} ~ {df_train['date'].max().date()} ({len(df_train)} rows)")
    print(f"Test Data Period: {df_test['date'].min().date()} ~ {df_test['date'].max().date()} ({len(df_test)} rows)")
    print(f"Model Input Dimension: {INPUT_DIM} (1 Sales + {len(meta_cols)} Meta Features)")

    # Scaling 
    sales_scaler = MinMaxScaler()
    df_train['sales_scaled'] = sales_scaler.fit_transform(df_train[['sales']].values)
    
    meta_train_vals = df_train[meta_cols].values
    
    # 시퀀스 구성: Sales + Meta 결합
    X_train, y_train = [], []
    sales_vals = df_train['sales_scaled'].values

    for i in range(len(df_train) - LOOKBACK - PREDICT + 1):
        sales_seq = sales_vals[i:i+LOOKBACK].reshape(-1, 1) # (L, 1)
        meta_seq = meta_train_vals[i:i+LOOKBACK]          # (L, 15)
        
        # 입력 X: Sales + Meta 결합 (L, 16)
        X_train.append(np.hstack([sales_seq, meta_seq])) 
        
        # 출력 Y: Sales (7일치)
        y_train.append(sales_vals[i+LOOKBACK:i+LOOKBACK+PREDICT])

    X_train = torch.tensor(np.array(X_train)).float().to(DEVICE) # (N, L, 16)
    y_train = torch.tensor(np.array(y_train)).float().to(DEVICE) # (N, 7)

    model = SimpleLSTMWithMeta(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Training LSTM (Meta Full)"):
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # --- 예측 (Predicting) - 순수 재귀적 예측 (Pure Recursive Forecasting) ---
    model.eval()
    test_predictions = []

    # 1. 초기 시퀀스 설정: 훈련 데이터의 마지막 LOOKBACK일 실제 Sales 및 Meta 값
    sales_full_scaled = sales_scaler.transform(df_train[['sales']].values)
    start_index = len(df_train) - LOOKBACK
    
    # current_sales_seq: 예측 값으로 갱신될 Sales 시퀀스
    current_sales_seq = sales_full_scaled[start_index :].squeeze() 
    
    # current_meta_seq: 훈련 데이터의 마지막 LOOKBACK일 Meta 값
    current_meta_seq = df_train[meta_cols].iloc[start_index:].values 

    # 2. 재귀적 예측 루프 시작 (len(df_test)만큼 예측)
    for t in tqdm(range(len(df_test)), desc="Recursive Prediction with Full Meta"):
        
        # 3. 모델 입력 구성 (Sales + Meta 결합)
        sales_input = current_sales_seq.reshape(-1, 1)
        x_t_input = np.hstack([sales_input, current_meta_seq])
        
        x_t = torch.tensor(x_t_input).float().to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(x_t).cpu().numpy().squeeze()
            
        # 4. 예측 값 추출 및 일요일 강제 0원 처리
        next_pred_scaled = pred_scaled[0] # 다음 날 예측 값 (스케일링 됨)
        restored_val = sales_scaler.inverse_transform([[next_pred_scaled]])[0, 0]
        
        # 일요일 확인 및 강제 0원 처리
        is_sunday = df_test['DOW_6'].iloc[t] == 1.0 
        
        # 일요일이면 0.0을 할당, 아니면 예측값 중 양수만 사용
        final_pred_val = 0.0 if is_sunday else max(0, restored_val) 

        test_predictions.append(final_pred_val)

        # 5. 다음 예측을 위한 시퀀스 업데이트 (Recursive Step)
        # Sales 시퀀스 갱신: 가장 오래된 값 제거, 모델 예측 값(스케일링된) 추가
        current_sales_seq = np.roll(current_sales_seq, shift=-1)
        current_sales_seq[-1] = next_pred_scaled

        # Meta 시퀀스 갱신: 가장 오래된 값 제거, 테스트 데이터의 다음 날 Meta 값 추가
        if t < len(df_test):
            next_meta_val = df_test[meta_cols].iloc[t].values
            current_meta_seq = np.roll(current_meta_seq, shift=-1, axis=0)
            current_meta_seq[-1, :] = next_meta_val
        
    # --- 최종 검증 및 시각화 ---
    y_true_test = df_test['sales'].values
    y_pred_test = np.array(test_predictions[:len(y_true_test)])

    test_mae = mean_absolute_error(y_true_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    test_smape = smape(y_true_test, y_pred_test)

    print("\n" + "="*50)
    print("📈 LSTM 성능 검증 (4. Sales + Full Meta - Pure Recursive)")
    print(f"Validation Period: {df_test['date'].min().date()} ~ {df_test['date'].max().date()}")
    print("="*50)
    print(f"1. MAE: {test_mae:,.2f} KRW")
    print(f"2. RMSE: {test_rmse:,.2f} KRW")
    print(f"3. SMAPE: {test_smape:.2f} %")
    print("="*50)

    # 시각화 (영어 레이블 사용)
    df_results = pd.DataFrame({'Date': df_test['date'].values, 'Actual_Sales': y_true_test, 'LSTM_Prediction': y_pred_test})
    plt.figure(figsize=(16, 6))
    plt.plot(df_results['Date'], df_results['Actual_Sales'], label='Actual Daily Sales', color='blue')
    plt.plot(df_results['Date'], df_results['LSTM_Prediction'], label='LSTM Pure Recursive Prediction (Full Meta)', color='red', linestyle='--')
    plt.title('4. LSTM Prediction (Sales + Full Meta - Pure Recursive) vs. Actual Daily Sales', fontsize=18)
    plt.xlabel('Date'); plt.ylabel('Daily Sales (KRW)'); plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_predict_validate()