import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ensemble_models에서 앙상블용 데이터/함수 가져오기
from ensemble_models import build_blend_model, smape


def main():
    # 1) ensemble_models에서 base_df, blend_model 가져오기
    #    (여기서 XGB/LGBM/LSTM/GRU + Blend 예측까지 다 계산됨)
    base_df, blend_model = build_blend_model()

    # 2) 기본 4모델 + 앙상블 예측 벡터 정리
    y_true = base_df["actual_daily"].values

    preds_dict = {
        "XGB":      base_df["pred_xgb"].values,
        "LGBM":     base_df["pred_lgbm"].values,
        "LSTM":     base_df["pred_lstm"].values,
        "GRU_ens":  base_df["pred_gru"].values,
        "Blend":    base_df["pred_blend"].values,
    }

    # 3) 모델별 MAE / RMSE / SMAPE 계산
    rows = []
    for name, pred in preds_dict.items():
        mae = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        s = smape(y_true, pred)
        rows.append({"model": name, "MAE": mae, "RMSE": rmse, "SMAPE": s})

    metrics_df = pd.DataFrame(rows)

    print("\n===== 기본 4모델 + 앙상블 성능 비교 =====")
    print(metrics_df.to_string(index=False))

    # 4) 바 차트로 성능 비교 (MAE / RMSE / SMAPE)
    models = metrics_df["model"].tolist()
    mae_vals = metrics_df["MAE"].tolist()
    rmse_vals = metrics_df["RMSE"].tolist()
    smape_vals = metrics_df["SMAPE"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Base Models vs Ensemble Performance", fontsize=14)

    axes[0].bar(models, mae_vals)
    axes[0].set_title("MAE")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(models, rmse_vals)
    axes[1].set_title("RMSE")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(models, smape_vals)
    axes[2].set_title("SMAPE (%)")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # 5) 타임시리즈: 실제 vs 4모델 + 앙상블 예측
    plt.figure(figsize=(18, 6))
    plt.plot(base_df["date"], base_df["actual_daily"],
             label="Actual", linewidth=2, color="black")

    plt.plot(base_df["date"], base_df["pred_xgb"],   label="XGB",      linestyle="--")
    plt.plot(base_df["date"], base_df["pred_lgbm"],  label="LGBM",     linestyle="--")
    plt.plot(base_df["date"], base_df["pred_lstm"],  label="LSTM",     linestyle="--")
    plt.plot(base_df["date"], base_df["pred_gru"],   label="GRU_ens",  linestyle="--")
    plt.plot(base_df["date"], base_df["pred_blend"], label="Blend",    linewidth=2)

    plt.title("Actual vs Base Models + Ensemble Predictions")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
