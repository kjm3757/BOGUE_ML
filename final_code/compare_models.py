import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from XGB_tuning import run_xgb
from LGBM_tuning import run_lgbm
from LSTM_final import run_lstm
from GRU_final import run_gru

# ------------------------------
# 공통 지표 함수 (SMAPE)
# ------------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom[denom == 0] = 1e-8
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def main():
    results = []
    test_dfs = {}

    # 1) 각 모델 실행
    print("🚀 XGB 실행 중...")
    xgb_metrics, xgb_df = run_xgb()
    results.append(xgb_metrics)
    test_dfs["XGB"] = xgb_df

    print("🚀 LGBM 실행 중...")
    lgbm_metrics, lgbm_df = run_lgbm()
    results.append(lgbm_metrics)
    test_dfs["LGBM"] = lgbm_df

    print("🚀 LSTM 실행 중...")
    lstm_metrics, lstm_df = run_lstm()
    results.append(lstm_metrics)
    test_dfs["LSTM"] = lstm_df

    print("🚀 GRU 실행 중...")
    gru_metrics, gru_df = run_gru()
    results.append(gru_metrics)
    test_dfs["GRU_ensemble"] = gru_df

    # 2) 메트릭 비교 표 출력
    metrics_df = pd.DataFrame(results)
    print("\n===== 모델별 성능 비교 =====")
    print(metrics_df.to_string(index=False))

    # 3) 예측 결과 컬럼 이름 통일
    # XGB / LGBM / LSTM: date, actual_daily, pred_daily (라고 가정)
    xgb_df = test_dfs["XGB"].rename(columns={"pred_daily": "pred_xgb"})
    lgbm_df = test_dfs["LGBM"].rename(columns={"pred_daily": "pred_lgbm"})
    lstm_df = test_dfs["LSTM"].rename(columns={"pred_daily": "pred_lstm"})

    # GRU: date, daily_actual, 예측매출 (라고 가정)
    gru_df = test_dfs["GRU_ensemble"].rename(
        columns={"daily_actual": "actual_daily", "예측매출": "pred_gru"}
    )

    # 4) 날짜 기준으로 inner join 해서 공통 구간만 사용
    base = xgb_df[["date", "actual_daily", "pred_xgb"]].copy()
    base = base.merge(lgbm_df[["date", "pred_lgbm"]], on="date", how="inner")
    base = base.merge(lstm_df[["date", "pred_lstm"]], on="date", how="inner")
    base = base.merge(gru_df[["date", "pred_gru"]], on="date", how="inner")
    base = base.sort_values("date").reset_index(drop=True)

    print("\n===== 공통 구간 예측 데이터 (앞부분 5줄) =====")
    print(base.head().to_string(index=False))

    # 5) 모델별 MAE / RMSE / SMAPE 막대 그래프
    models = metrics_df["model"].tolist()
    mae_vals = metrics_df["test_MAE"].tolist()
    rmse_vals = metrics_df["test_RMSE"].tolist()
    smape_vals = metrics_df["test_SMAPE"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Model-wise Performance Comparison", fontsize=14)

    # MAE
    axes[0].bar(models, mae_vals)
    axes[0].set_title("MAE")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=45)

    # RMSE
    axes[1].bar(models, rmse_vals)
    axes[1].set_title("RMSE")
    axes[1].tick_params(axis="x", rotation=45)

    # SMAPE
    axes[2].bar(models, smape_vals)
    axes[2].set_title("SMAPE (%)")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # 6) 타임시리즈: 실제 vs 4개 모델 예측
    plot_df = base.copy()

    plt.figure(figsize=(18, 6))
    plt.plot(plot_df["date"], plot_df["actual_daily"], label="Actual", linewidth=2)

    plt.plot(plot_df["date"], plot_df["pred_xgb"],   label="XGB",        linestyle="--")
    plt.plot(plot_df["date"], plot_df["pred_lgbm"],  label="LGBM",       linestyle="--")
    plt.plot(plot_df["date"], plot_df["pred_lstm"],  label="LSTM",       linestyle="--")
    plt.plot(plot_df["date"], plot_df["pred_gru"],   label="GRU_ens",    linestyle="--")

    plt.title("Daily Sales: Actual vs Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 7) (옵션) 4모델 단순 평균 앙상블 성능도 같이 출력
    y_true = plot_df["actual_daily"].values
    preds = plot_df[["pred_xgb", "pred_lgbm", "pred_lstm", "pred_gru"]].values
    pred_mean4 = preds.mean(axis=1)

    mean4_mae = np.mean(np.abs(y_true - pred_mean4))
    mean4_rmse = np.sqrt(np.mean((y_true - pred_mean4) ** 2))
    mean4_smape = smape(y_true, pred_mean4)

    print("\n===== 단순 평균 앙상블 (4모델 평균) 성능 =====")
    print(f"MAE   : {mean4_mae:,.2f}")
    print(f"RMSE  : {mean4_rmse:,.2f}")
    print(f"SMAPE : {mean4_smape:.2f} %")


if __name__ == "__main__":
    main()
