import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os

# =========================================================
# 0) Seed 고정 함수 (매번 다른 시드 적용 예정)
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 1) 평가 지표 함수
# =========================================================
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100
    return mae, rmse, smape

# =========================================================
# 2) 데이터 전처리
# =========================================================
def clean_sales(df):
    if df["daily"].dtype == "object":
        df["daily"] = (
            df["daily"]
            .astype(str)
            .str.replace(",", "")
            .str.replace(" ", "")
            .str.strip()
        )
    df["daily"] = pd.to_numeric(df["daily"], errors="coerce").fillna(0)
    return df


def add_date_features(df):
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    # 운영 시간 (단순 규칙)
    df["open_hours"] = 11
    df.loc[df["weekday"] == 5, "open_hours"] = 6  # 토요일
    df.loc[df["weekday"] == 6, "open_hours"] = 0  # 일요일
    return df


def add_lag_features(df):
    df = df.copy()
    df["lag1"] = df["daily"].shift(1)
    df["lag7"] = df["daily"].shift(7)
    df["lag14"] = df["daily"].shift(14)
    df["lag28"] = df["daily"].shift(28)

    df["roll_mean7"] = df["daily"].rolling(7).mean()
    df["roll_mean14"] = df["daily"].rolling(14).mean()
    df["roll_mean28"] = df["daily"].rolling(28).mean()

    df["roll_std7"] = df["daily"].rolling(7).std()
    df["roll_std28"] = df["daily"].rolling(28).std()
    return df

# =========================================================
# 3) Sliding Window
# =========================================================
def create_sequences(df, feature_cols, target_col, seq_len=28):
    X, y = [], []
    feature_vals = df[feature_cols].values
    target_vals = df[target_col].values

    for i in range(seq_len, len(df)):
        seq_x = feature_vals[i - seq_len + 1 : i + 1]
        if len(seq_x) == seq_len:
            X.append(seq_x)
            y.append(target_vals[i])
    return np.array(X), np.array(y)

# =========================================================
# 4) Dataset & Model
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.relu(out[:, -1, :]))

# =========================================================
# 5) 유연한 날짜 예측 함수 (순수 예측값 반환)
# =========================================================
def flexible_forecast_raw(
    model,
    df_history,
    df_future_meta,
    start_date,
    end_date,
    seq_len,
    feature_cols,
    scaler_X,
    scaler_y,
):
    model.eval()
    target_dates = pd.date_range(start=start_date, end=end_date)

    history = df_history.copy()
    preds = []

    for date in target_dates:
        if date not in df_future_meta["date"].values:
            row = pd.Series(0, index=df_future_meta.columns)
            row["date"] = date
        else:
            row = df_future_meta[df_future_meta["date"] == date].iloc[0].copy()

        last_vals = history["daily"].values

        row["lag1"] = last_vals[-1]
        row["lag7"] = last_vals[-7] if len(last_vals) >= 7 else 0
        row["lag14"] = last_vals[-14] if len(last_vals) >= 14 else 0
        row["lag28"] = last_vals[-28] if len(last_vals) >= 28 else 0

        row["roll_mean7"] = (
            pd.Series(last_vals[-7:]).mean() if len(last_vals) >= 7 else 0
        )
        row["roll_mean14"] = (
            pd.Series(last_vals[-14:]).mean() if len(last_vals) >= 14 else 0
        )
        row["roll_mean28"] = (
            pd.Series(last_vals[-28:]).mean() if len(last_vals) >= 28 else 0
        )

        row["roll_std7"] = pd.Series(last_vals[-7:]).std() if len(last_vals) >= 7 else 0
        row["roll_std28"] = (
            pd.Series(last_vals[-28:]).std() if len(last_vals) >= 28 else 0
        )

        row_df = pd.DataFrame([row])
        row_df = add_date_features(row_df)
        row = row_df.iloc[0].copy()

        temp_history = pd.concat(
            [history.tail(seq_len - 1), row_df], ignore_index=True
        )
        seq_unscaled = temp_history[feature_cols]
        seq_scaled = scaler_X.transform(seq_unscaled)

        X = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(X).item()

        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        pred = max(pred, 0)

        preds.append(pred)

        row["daily"] = pred
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    return preds

# =========================================================
# 6) 메인 실행부
# =========================================================
if __name__ == "__main__":
    # ---- 경로 설정 (현재 파일 기준) ----
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "Data")

    TRAIN_PATH = os.path.join(DATA_DIR, "POS_train_val.csv")
    TEST_PATH = os.path.join(DATA_DIR, "POS_test.csv")
    ACAD_PATH = os.path.join(DATA_DIR, "Feature.xlsx")

    print("📂 데이터 로드 중...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    academic = pd.read_excel(ACAD_PATH)  # ✅ 엑셀 파일은 read_excel

    # ---- 컬럼: date / daily 로 가정 ----
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    academic["date"] = pd.to_datetime(academic["date"])

    train = clean_sales(train)
    test = clean_sales(test)

    # 학사 캘린더 처리
    weekday_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    if "weekday" in academic.columns:
        academic["acad_weekday"] = academic["weekday"].map(weekday_map)
        academic = academic.drop(columns=["weekday"])

    academic = academic.rename(
        columns={
            "weekend": "acad_weekend",
            "holiday": "acad_holiday",
            "semester": "acad_semester",
            "seasonal": "acad_seasonal",
            "exam": "acad_exam",
            "ceremony": "acad_ceremony",
            "dormitory": "acad_dormitory",
        }
    )

    # ---- Merge ----
    train = train.merge(academic, on="date", how="left")
    test_meta = test.merge(academic, on="date", how="left")

    # ---- 특징 생성 ----
    train = add_date_features(train)
    test_meta = add_date_features(test_meta)

    train = add_lag_features(train).dropna().reset_index(drop=True)

    feature_cols = [
        "acad_weekend",
        "acad_semester",
        "acad_weekday",
        "open_hours",
        "acad_ceremony",
        "acad_exam",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "lag1",
        "lag7",
        "lag14",
        "lag28",
        "roll_std7",
        "roll_std28",
        "roll_mean7",
        "roll_mean14",
        "roll_mean28",
    ]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_scaled = train.copy()
    train_scaled[feature_cols] = scaler_X.fit_transform(train[feature_cols])
    train_scaled[["daily"]] = scaler_y.fit_transform(train[["daily"]])

    seq_len = 28
    X_train, y_train = create_sequences(train_scaled, feature_cols, "daily", seq_len)
    dataset = SequenceDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # =========================================================
    # 🔥 Seed Ensemble (5개 모델 학습)
    # =========================================================
    SEEDS = [42, 100, 2024, 777, 999]
    all_preds = []

    print(f"\n🚀 Seed Ensemble 시작 (총 {len(SEEDS)}개 모델 학습)...")

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i + 1}/{len(SEEDS)}] Seed {seed} 학습 중...")
        set_seed(seed)

        model = GRUModel(
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
        ).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(100):  # Epoch 100
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                output = model(Xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

        # ---- 예측 구간 설정 ----
        PREDICT_START_DATE = "2025-08-10"
        PREDICT_END_DATE = "2025-10-31"

        preds = flexible_forecast_raw(
            model,
            train,
            test_meta,
            PREDICT_START_DATE,
            PREDICT_END_DATE,
            seq_len,
            feature_cols,
            scaler_X,
            scaler_y,
        )
        all_preds.append(preds)

    # =========================================================
    # 🎯 앙상블 평균 및 후처리
    # =========================================================
    print("\n🎯 앙상블 결과 집계 중...")

    avg_preds = np.mean(all_preds, axis=0)
    final_preds = []
    target_dates = pd.date_range(start=PREDICT_START_DATE, end=PREDICT_END_DATE)

    for i, date in enumerate(target_dates):
        pred = avg_preds[i]

        if date in test_meta["date"].values:
            row = test_meta[test_meta["date"] == date].iloc[0]

            is_sunday = row["date"].weekday() == 6
            is_closed = row.get("open_hours", 11) == 0
            is_holiday = row.get("acad_holiday", 0) == 1

            if is_sunday or is_closed or is_holiday:
                pred = 0

        if pred < 10000:
            pred = 0

        final_preds.append(pred)

    forecast_df = pd.DataFrame({"date": target_dates, "예측매출": final_preds})

    # =========================================================
    # 📊 성능 평가
    # =========================================================
    actual_df = test[["date", "daily"]].copy()
    actual_df.rename(columns={"daily": "daily_actual"}, inplace=True)

    pred_df = forecast_df[["date", "예측매출"]].copy()
    comparison_df = pd.merge(actual_df, pred_df, on="date", how="inner")

    if not comparison_df.empty:
        mae, rmse, smape_val = calculate_metrics(
            comparison_df["daily_actual"], comparison_df["예측매출"]
        )

        print("\n" + "=" * 40)
        print(f"📊 성능 평가 결과 (비교 데이터: {len(comparison_df)}개)")
        print("=" * 40)
        print(f"1. MAE   : {mae:,.2f}")
        print(f"2. RMSE  : {rmse:,.2f}")
        print(f"3. SMAPE : {smape_val:.2f} %")
        print("=" * 40)

        result_dir = os.path.join(BASE_DIR, "..", "result")
        os.makedirs(result_dir, exist_ok=True)

        out_path = os.path.join(result_dir, "forecast_comparison_result_ensemble.csv")
        comparison_df.to_csv(out_path, index=False)

        print(f"\n✅ 비교 결과 저장 완료: {out_path}")
    else:
        print("\n⚠️ 예측 구간과 실제 Test 데이터 구간이 겹치지 않습니다.")
