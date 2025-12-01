import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 📌 1. 환경 설정 및 지표 정의
# --------------------------------------------------------------------------
set_seed = lambda x: np.random.seed(x) or torch.manual_seed(x)
set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 32, 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📌 CSV 파일 경로 정의
TRAIN_CSV = 'POS_train_val.csv'
TEST_CSV = 'POS_test.csv'

SALES_COL = '일매출'
DATE_COL = '영업일자'
GROUP_COL = '그룹키'

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 1e-6
    return np.mean(numerator / denominator) * 100

# --- 2. LSTM 모델 클래스 정의 (Sales Only) ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 3. 데이터 로드 및 특징 생성 함수 (CSV 파일 기준으로 Train/Test 분리) ---
def create_data_for_lstm(train_csv, test_csv):
    try:
        df_train_raw = pd.read_csv(train_csv)
        df_test_raw = pd.read_csv(test_csv)
    except Exception as e:
        print(f"Error: File loading failed. {e}"); return pd.DataFrame(), pd.DataFrame(), []

    def clean_data(df):
        # 컬럼 이름 정리
        df.rename(columns={DATE_COL: 'date', SALES_COL: 'sales'}, inplace=True)
        
        # 날짜 및 숫자 전처리
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        def clean_sales(series):
            # 쉼표(,) 제거 및 float 변환
            series = series.astype(str).str.replace(',', '', regex=False)
            return pd.to_numeric(series, errors='coerce').fillna(0)
        
        df['sales'] = clean_sales(df['sales'])
        df[GROUP_COL] = '전체'
        return df.sort_values('date').fillna(0).reset_index(drop=True)

    df_train = clean_data(df_train_raw)
    df_test = clean_data(df_test_raw)
    
    meta_cols = []

    # df_train과 df_test를 분리하여 반환
    return df_train, df_test, meta_cols

# --- 4. 훈련, 예측 및 검증 함수 ---
def train_predict_validate():
    # 📌 변경: df_train과 df_test를 파일 기준으로 분리하여 로드
    df_train, df_test, meta_cols = create_data_for_lstm(TRAIN_CSV, TEST_CSV)
    if df_train.empty or df_test.empty: return

    print(f"Train Data Period: {df_train['date'].min().date()} ~ {df_train['date'].max().date()} ({len(df_train)} rows)")
    print(f"Test Data Period: {df_test['date'].min().date()} ~ {df_test['date'].max().date()} ({len(df_test)} rows)")

    # Scaling (df_train만 사용)
    sales_scaler = MinMaxScaler()
    df_train['sales_scaled'] = sales_scaler.fit_transform(df_train[['sales']].values)

    # 시퀀스 구성
    X_train, y_train = [], []
    sales_vals = df_train['sales_scaled'].values

    for i in range(len(df_train) - LOOKBACK - PREDICT + 1):
        X_train.append(sales_vals[i:i+LOOKBACK])
        y_train.append(sales_vals[i+LOOKBACK:i+LOOKBACK+PREDICT])

    X_train = torch.tensor(np.array(X_train)).float().to(DEVICE).unsqueeze(-1) # (N, L, 1)
    y_train = torch.tensor(np.array(y_train)).float().to(DEVICE)

    model = SimpleLSTM(input_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Training LSTM (Minimal)"):
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

    # 📌 1. 초기 시퀀스 설정: 훈련 데이터의 마지막 LOOKBACK일 실제 값
    sales_full_scaled = sales_scaler.transform(df_train[['sales']].values)
    start_index = len(df_train) - LOOKBACK
    
    # current_sales_seq: 예측 값으로 갱신될 Sales 시퀀스 (실제 값으로 시작)
    current_sales_seq = sales_full_scaled[start_index :].squeeze().copy()

    # 📌 2. 재귀적 예측 루프 시작 (len(df_test)만큼 예측)
    for t in tqdm(range(len(df_test)), desc="Recursive Prediction"):
        
        # 3. 모델 입력 구성 (Sales Only): (1, LOOKBACK, 1)
        x_t = torch.tensor(current_sales_seq).float().to(DEVICE).unsqueeze(0).unsqueeze(-1) 

        with torch.no_grad():
            # 모델은 7일치 예측을 하지만, 재귀적 예측을 위해 첫 1일치만 사용
            pred_scaled = model(x_t).cpu().numpy().squeeze()
            
        # 4. 예측 값 추출 및 양수 처리 (미래 정보 사용 제거)
        next_pred_scaled = pred_scaled[0] # 다음 날 예측 값 (스케일링 됨)
        
        # 스케일링 복원
        restored_val = sales_scaler.inverse_transform([[next_pred_scaled]])[0, 0]
        
        final_pred_val = max(0, restored_val) 

        test_predictions.append(final_pred_val)

        # 5. 다음 예측을 위한 시퀀스 업데이트 (Recursive Step)
        # Sales 시퀀스 갱신: 가장 오래된 값 제거, 모델 예측 값(스케일링된) 추가
        current_sales_seq = np.roll(current_sales_seq, shift=-1)
        current_sales_seq[-1] = next_pred_scaled

    # --- 최종 검증 및 시각화 ---
    y_true_test = df_test['sales'].values
    y_pred_test = np.array(test_predictions[:len(y_true_test)])

    test_mae = mean_absolute_error(y_true_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    test_smape = smape(y_true_test, y_pred_test)

    print("\n" + "="*50)
    print("📈 LSTM 성능 검증 (2. Sales Only - Pure Recursive)")
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
    plt.plot(df_results['Date'], df_results['LSTM_Prediction'], label='LSTM Pure Recursive Prediction', color='red', linestyle='--')
    plt.title('2. LSTM Prediction (Sales Only - Pure Recursive) vs. Actual Daily Sales (No Future Info)', fontsize=18)
    plt.xlabel('Date'); plt.ylabel('Daily Sales (KRW)'); plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_predict_validate()
