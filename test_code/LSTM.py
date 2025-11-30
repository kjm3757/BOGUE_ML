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
SALES_FILE = '월_매출 (2).xlsx'
CALENDAR_FILE = '학사일정_정리(2325) (3).xlsx'
TRAIN_END_DATE = pd.to_datetime('2025-04-30')
SALES_COL = '일매출'
GROUP_COL = '그룹키'

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 1e-6
    return np.mean(numerator / denominator) * 100

# --- 2. LSTM 모델 클래스 정의 (Meta 기능 포함) ---
class SimpleLSTMWithMeta(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 3. 데이터 로드 및 특징 생성 함수 (Full Features) ---
def create_data_for_lstm(sales_file, calendar_file):
    try:
        df_sales = pd.read_excel(sales_file)
        df_calendar = pd.read_excel(calendar_file)
    except Exception as e:
        print(f"Error: File loading failed. {e}"); return pd.DataFrame(), []

    df_sales.rename(columns={'영업일자': 'date', SALES_COL: 'sales'}, inplace=True)
    df_calendar.rename(columns={'date': 'date'}, inplace=True)
    df_sales['date'] = pd.to_datetime(df_sales['date'], errors='coerce')
    df_calendar['date'] = pd.to_datetime(df_calendar['date'], errors='coerce')

    # FIX: 학사일정 파일에서 문자열 컬럼 제거
    cols_to_drop = [c for c in ['weekday', 'semester', 'ceremony'] if c in df_calendar.columns]
    if cols_to_drop: df_calendar = df_calendar.drop(columns=cols_to_drop, errors='ignore')

    df_merged = pd.merge(df_sales, df_calendar, on='date', how='left')
    df_merged[GROUP_COL] = '전체'
    df_merged = df_merged.sort_values('date').fillna(0)
    df_merged['sales'] = df_merged['sales'].astype(float)
    df_merged['DayOfWeek'] = df_merged['date'].dt.dayofweek

    out = df_merged.copy()

    # 📌 2. 시간 및 주기 특징
    out['Month'] = out['date'].dt.month
    out['Day'] = out['date'].dt.day
    out['OpHoursFactor'] = out['DayOfWeek'].apply(lambda x: 1.0 if x < 5 else (0.55 if x == 5 else 0.0))
    out['IsPublicHoliday'] = out['holiday'].astype(float)
    out['IsExamPeriod'] = out['exam'].astype(float)
    out['IsDormitoryEvent'] = out['dormitory'].astype(float)

    out = out.fillna(0)

    # 📌 특징 목록: 순수하게 미리 알려진 특징만 포함
    exclude = ['date', 'sales', GROUP_COL, 'weekday', 'semester', 'ceremony', 'weekend']

    # 매출 기반의 특징 목록을 정의합니다.
    sales_based_features = [f'Lag{l}' for l in [1, 2, 3, 7, 14, 28]] + \
                          [f'RollingMean{w}' for w in [7, 14, 28]] + \
                          [f'RollingStd{w}' for w in [7, 14, 28]]

    # 순수 Meta 특징만 선택 (Lag/Rolling을 제외한 나머지)
    meta_cols = [col for col in out.columns
                if col not in exclude
                and col not in sales_based_features
                and np.issubdtype(out[col].dtype, np.number)]

    return out, meta_cols

# --- 4. 훈련, 예측 및 검증 함수 ---
def train_predict_validate():
    df_full, meta_cols = create_data_for_lstm(SALES_FILE, CALENDAR_FILE)
    if df_full.empty: return

    df_train = df_full[df_full['date'] <= TRAIN_END_DATE].copy()
    df_test = df_full[df_full['date'] > TRAIN_END_DATE].copy()
    df_history = df_full.sort_values('date').reset_index(drop=True)

    # Scaling
    sales_scaler = MinMaxScaler()
    meta_scaler = MinMaxScaler()
    df_train['sales_scaled'] = sales_scaler.fit_transform(df_train[['sales']].values)
    df_train[meta_cols] = meta_scaler.fit_transform(df_train[meta_cols].values)

    # 시퀀스 구성 및 학습 (생략)
    X_train, y_train, M_train = [], [], []
    sales_vals = df_train['sales_scaled'].values
    meta_vals = df_train[meta_cols].values

    for i in range(len(df_train) - LOOKBACK - PREDICT + 1):
        X_train.append(sales_vals[i:i+LOOKBACK])
        M_train.append(meta_vals[i:i+LOOKBACK])
        y_train.append(sales_vals[i+LOOKBACK:i+LOOKBACK+PREDICT])

    X_train = torch.tensor(np.array(X_train)).float().to(DEVICE).unsqueeze(-1)
    M_train = torch.tensor(np.array(M_train)).float().to(DEVICE)
    y_train = torch.tensor(np.array(y_train)).float().to(DEVICE)
    LSTM_Input = torch.cat([X_train, M_train], dim=-1)

    model = SimpleLSTMWithMeta(input_dim=LSTM_Input.size(-1)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Training LSTM (Full)"):
        idx = torch.randperm(len(LSTM_Input))
        for i in range(0, len(LSTM_Input), BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            X_batch, y_batch = LSTM_Input[batch_idx], y_train[batch_idx]
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # --- 예측 (Predicting) - 재귀적 예측 (Recursive Forecasting) ---
    model.eval()
    test_predictions = []

    # 📌 1. 전체 데이터 스케일링 (초기 시퀀스 및 메타 데이터 추출에 사용)
    sales_full_scaled = sales_scaler.transform(df_history[['sales']].values)
    meta_full_scaled = meta_scaler.transform(df_history[meta_cols].values)
    start_index_history = len(df_train) # 테스트 시작일 인덱스

    # 📌 2. 재귀적 예측을 위한 초기 시퀀스 설정 (UnboundLocalError 해결)
    start_index = len(df_train) - LOOKBACK
    current_sales_seq = sales_full_scaled[start_index : start_index + LOOKBACK].squeeze().copy()
    current_meta_seq = meta_full_scaled[start_index : start_index + LOOKBACK].copy()

    # 테스트 기간의 메타 데이터만 추출 (T_end+1일 부터)
    meta_test_vals_scaled = meta_full_scaled[start_index_history:].copy()

    # 📌 3. 재귀적 예측 루프 시작
    for t in tqdm(range(len(df_test)), desc="Recursive Prediction"):

        # 3. 모델 입력 구성 (현재 시퀀스 사용)
        x_t = torch.tensor(current_sales_seq).float().to(DEVICE).unsqueeze(0).unsqueeze(-1)
        m_t = torch.tensor(current_meta_seq).float().to(DEVICE).unsqueeze(0)
        lstm_input = torch.cat([x_t, m_t], dim=-1)

        with torch.no_grad():
            pred_scaled = model(lstm_input).cpu().numpy().squeeze()

        # 4. 예측 값 추출 및 일요일 처리
        next_pred_scaled = pred_scaled[0] # 다음 날 예측 값 (스케일링 됨)
        restored_val = sales_scaler.inverse_transform([[next_pred_scaled]])[0, 0]

        is_sunday = df_test['date'].dt.dayofweek.iloc[t] == 6
        final_pred_val = max(0, restored_val) if not is_sunday else 0

        test_predictions.append(final_pred_val)

        # 5. 다음 예측을 위한 시퀀스 업데이트 (Recursive Step)

        # Sales 시퀀스 갱신: 가장 오래된 값 제거, 모델 예측 값(스케일링된) 추가
        current_sales_seq = np.roll(current_sales_seq, shift=-1)
        current_sales_seq[-1] = next_pred_scaled

        # Meta 시퀀스 갱신: 가장 오래된 값 제거, 다음 날 메타 데이터 추가
        if t < len(df_test) - 1:
            next_day_meta = meta_test_vals_scaled[t]
            current_meta_seq = np.roll(current_meta_seq, shift=-1, axis=0)
            current_meta_seq[-1] = next_day_meta

    # --- 최종 검증 및 시각화 ---
    y_true_test = df_test['sales'].values
    y_pred_test = np.array(test_predictions[:len(y_true_test)])

    test_mae = mean_absolute_error(y_true_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    test_smape = smape(y_true_test, y_pred_test)

    print("\n" + "="*50)
    print("📈 LSTM 성능 검증 (재귀적 예측)")
    print(f"1. MAE: {test_mae:,.2f} 원")
    print(f"2. RMSE: {test_rmse:,.2f} 원")
    print(f"3. SMAPE: {test_smape:.2f} %")
    print("="*50)

    # 시각화
    df_results = pd.DataFrame({'Date': df_test['date'].values, 'Actual_Sales': y_true_test, 'LSTM_Prediction': y_pred_test})
    plt.figure(figsize=(16, 6))
    plt.plot(df_results['Date'], df_results['Actual_Sales'], label='Actual Daily Sales', color='blue')
    plt.plot(df_results['Date'], df_results['LSTM_Prediction'], label='LSTM Recursive Prediction', color='red', linestyle='--')
    plt.title('LSTM Recursive Prediction vs. Actual Daily Sales', fontsize=18)
    plt.xlabel('Date'); plt.ylabel('Daily Sales (KRW)'); plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_predict_validate()