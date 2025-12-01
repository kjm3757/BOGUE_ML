import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# 🔽 로컬에서 각 모델 실행 함수 import
from XGB_tuning import run_xgb
from LGBM_tuning import run_lgbm
from LSTM_final import run_lstm
from GRU_final import run_gru


# ==========================================================
# 🔧 공통 함수: SMAPE
# ==========================================================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    # (|y_true| + |y_pred|) / 2  형태로 괄호만 살짝 수정
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom[denom == 0] = 1e-8
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


# ==========================================================
# 🧠 전체 파이프라인 (4모델 실행 + 블렌딩)
# ==========================================================
def build_blend_model():
    # 1️⃣ 4개 모델 실행
    xgb_metrics, xgb_df = run_xgb()
    lgbm_metrics, lgbm_df = run_lgbm()
    lstm_metrics, lstm_df = run_lstm()
    gru_metrics, gru_df = run_gru()

    # 2️⃣ 컬럼 이름 통일
    xgb_df  = xgb_df.rename(columns={"pred_daily": "pred_xgb"})
    lgbm_df = lgbm_df.rename(columns={"pred_daily": "pred_lgbm"})
    lstm_df = lstm_df.rename(columns={"pred_daily": "pred_lstm"})
    gru_df  = gru_df.rename(columns={
        "daily_actual": "actual_daily",
        "예측매출": "pred_gru"
    })

    # 3️⃣ 공통 날짜 기준 inner join → 베이스 DataFrame 생성
    base = xgb_df[["date", "actual_daily", "pred_xgb"]].copy()
    base = base.merge(lgbm_df[["date", "pred_lgbm"]], on="date", how="inner")
    base = base.merge(lstm_df[["date", "pred_lstm"]], on="date", how="inner")
    base = base.merge(gru_df[["date", "pred_gru"]], on="date", how="inner")
    base = base.sort_values("date").reset_index(drop=True)

    print("\n===== 앙상블에 사용할 공통 구간 데이터 (샘플) =====")
    print(base.head())

    # 4️⃣ Linear Regression Blending 학습
    y_true = base["actual_daily"].values
    pred_matrix = base[["pred_xgb", "pred_lgbm", "pred_lstm", "pred_gru"]].values

    blender = LinearRegression()
    blender.fit(pred_matrix, y_true)
    pred_blend = blender.predict(pred_matrix)

    base["pred_blend"] = pred_blend

    # 5️⃣ 앙상블 성능 출력
    b_mae = mean_absolute_error(y_true, pred_blend)
    b_rmse = np.sqrt(mean_squared_error(y_true, pred_blend))
    b_smape = smape(y_true, pred_blend)

    print("\n===== [Linear Regression Blending] 최종 앙상블 성능 =====")
    print(f"MAE:   {b_mae:,.2f}")
    print(f"RMSE:  {b_rmse:,.2f}")
    print(f"SMAPE: {b_smape:.2f}%")

    # 6️⃣ Actual vs Ensemble Plot
    plt.figure(figsize=(18, 6))
    plt.plot(base["date"], base["actual_daily"], label="Actual", linewidth=2)
    plt.plot(base["date"], base["pred_blend"], label="Blend (Ensemble)", linestyle="--")
    plt.title("Actual vs Blended Ensemble Prediction")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return base, blender


# ==========================================================
# 🚀 실행
# ==========================================================
if __name__ == "__main__":
    base_df, blend_model = build_blend_model()

    print("\n===== Blending 모델 가중치 (Feature Importances) =====")
    for name, w in zip(["XGB", "LGBM", "LSTM", "GRU"], blend_model.coef_):
        print(f"{name:5s} : {w:.6f}")
    print(f"Intercept : {blend_model.intercept_:.6f}")
