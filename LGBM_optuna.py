import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import optuna


# =========================
# 1. 유틸 함수들
# =========================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100


def calculate_operating_hours(row):
    """
    학기/방학 + 요일 + 공휴일 기준 운영시간 계산
    """
    weekday = row["weekday"]
    semester = row["semester"]
    holiday = row["holiday"]

    # 일요일 미운영
    if weekday == "sun":
        return 0

    # 학기 중
    if semester == 1:
        if holiday == 1:
            return 7  # 공휴일 7시간
        if weekday in ["mon", "tue", "wed", "thu", "fri"]:
            return 12  # 평일 12시간
        if weekday == "sat":
            return 7
        return 0

    # 방학 중
    else:
        if holiday == 1:
            return 0
        if weekday in ["mon", "tue", "wed", "thu", "fri", "sat"]:
            return 7
        return 0


# =========================
# 2. 메인 파이프라인
# =========================
def main():
    FEATURE = "Data/Feature.xlsx"
    TRAIN = "Data/POS_train_val.csv"
    TEST = "Data/POS_test.csv"

    # ---------- (1) 데이터 로드 ----------
    print("Loading data...")
    feat = pd.read_excel(FEATURE)
    pos_train = pd.read_csv(TRAIN)
    pos_test = pd.read_csv(TEST)

    # 숫자 전처리
    for df in [pos_train, pos_test]:
        for col in ["daily", "AOV"]:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False).astype(float)
            )
        df["date"] = pd.to_datetime(df["date"])

    feat["date"] = pd.to_datetime(feat["date"])

    # train/test 구분 플래그 추가 후 concat
    pos_train["set"] = "train"
    pos_test["set"] = "test"
    pos_all = pd.concat([pos_train, pos_test], ignore_index=True)

    # Feature와 merge
    df_all = pd.merge(pos_all, feat, on="date", how="left")
    df_all = df_all.sort_values("date").reset_index(drop=True)

    print("All merged shape:", df_all.shape)
    print(df_all.head())

    # ---------- (2) Feature Engineering ----------
    print("\nFeature engineering...")

    # 2-1. 운영시간
    df_all["operating_hours"] = df_all.apply(calculate_operating_hours, axis=1)

    # 2-2. 시험 window (전후 3일 누적)
    df_all["exam_before3"] = (
        df_all["exam"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    )
    df_all["exam_after3"] = (
        df_all["exam"].shift(-1).rolling(3, min_periods=1).sum().fillna(0)
    )

    # 2-3. 학기 × 주말 교차
    df_all["semester_weekend"] = df_all["semester"] * df_all["weekend"]

    # 2-4. Lag Features (지연값)
    lag_list = [1, 2, 3, 7, 14, 28]
    for lag in lag_list:
        df_all[f"Lag{lag}"] = df_all["daily"].shift(lag)

    # 2-5. Rolling Mean / Std (이동 평균 / 표준편차)
    win_list = [7, 14, 28]
    for win in win_list:
        roll = df_all["daily"].rolling(window=win, min_periods=1)
        df_all[f"RollingMean{win}"] = roll.mean().shift(1)
        df_all[f"RollingStd{win}"] = roll.std(ddof=0).shift(1)

    # 2-6. weekday one-hot
    df_all = pd.get_dummies(df_all, columns=["weekday"], drop_first=True)

    print("\nAfter feature engineering:")
    print(df_all.head())

    # ---------- (3) Train / Test 분리 ----------
    feature_na_cols = [f"Lag{l}" for l in lag_list] + \
                      [f"RollingMean{w}" for w in win_list] + \
                      [f"RollingStd{w}" for w in win_list]

    df_all_clean = df_all.dropna(subset=feature_na_cols).reset_index(drop=True)
    print("\nAfter dropping NaN from lag/rolling:", df_all_clean.shape)

    df_train_val = df_all_clean[df_all_clean["set"] == "train"].copy()
    df_test = df_all_clean[df_all_clean["set"] == "test"].copy()

    # 학습은 매출 0원 제외
    df_train_val = df_train_val[df_train_val["daily"] > 0].copy()
    print("Train_val (daily>0):", df_train_val.shape)
    print("Test (all, incl. 0 & >0):", df_test.shape)

    # ---------- (4) Train/Val Split (시간 기준 80:20) ----------
    df_sorted = df_train_val.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train = df_sorted.iloc[:split_idx]
    val = df_sorted.iloc[split_idx:]

    drop_cols = ["date", "daily", "num", "AOV", "set"]
    feature_cols = [c for c in df_sorted.columns if c not in drop_cols]

    X_train = train[feature_cols]
    y_train = train["daily"]
    X_val = val[feature_cols]
    y_val = val["daily"]

    print("\n# of features:", len(feature_cols))
    print(feature_cols)

    # ---------- (5) Optuna Objective 정의 ----------
    def objective(trial: optuna.trial.Trial) -> float:
        # 하이퍼파라미터 후보 탐색 범위 설정
        params = {
            "objective": "regression",
            "random_state": 42,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }

        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train,
        )

        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        return rmse  # Optuna가 최소화할 값

    # ---------- (6) Optuna로 튜닝 ----------
    print("\nRunning Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    # n_trials을 늘리면 더 많이 탐색하지만 시간이 오래 걸림 (예: 50~100)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n=== Optuna best trial ===")
    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

    best_params = study.best_params
    best_params["objective"] = "regression"
    best_params["random_state"] = 42

    # ---------- (7) Best 파라미터로 Train+Val 전체 재학습 ----------
    X_train_full = df_sorted[feature_cols]
    y_train_full = df_sorted["daily"]

    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train_full, y_train_full)

    # 참고용: 같은 Val 구간에 대해 성능 다시 측정
    val_pred = best_model.predict(X_val)
    mae_val = mean_absolute_error(y_val, val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    smape_val = smape(y_val.values, val_pred)

    print("\n===== Validation Performance (Optuna-tuned LightGBM) =====")
    print(f"MAE   : {mae_val:,.2f}")
    print(f"RMSE  : {rmse_val:,.2f}")
    print(f"SMAPE : {smape_val:.2f}%")

    print("\nFeature Importances (Top 30):")
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(fi.head(30))

    # ---------- (8) Test 평가 (daily>0만) ----------
    df_test_eval = df_test[df_test["daily"] > 0].copy()
    if len(df_test_eval) == 0:
        print("\nNo test rows with daily>0 after lag/rolling/NaN filtering.")
        return

    X_test = df_test_eval[feature_cols]
    y_test = df_test_eval["daily"].values
    test_pred = best_model.predict(X_test)

    mae_test = mean_absolute_error(y_test, test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
    smape_test = smape(y_test, test_pred)

    print("\n===== Test Performance (daily>0 only, Optuna-tuned LightGBM) =====")
    print(f"MAE   : {mae_test:,.2f}")
    print(f"RMSE  : {rmse_test:,.2f}")
    print(f"SMAPE : {smape_test:.2f}%")

    result_df = pd.DataFrame({
        "date": df_test_eval["date"].values,
        "actual_daily": y_test,
        "pred_daily": test_pred
    }).sort_values("date").reset_index(drop=True)

    print("\nTest prediction head:")
    print(result_df.head(20))


if __name__ == "__main__":
    main()
