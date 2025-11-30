import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb  # ✅ LightGBM 대신 XGBoost 사용

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
    FEATURE = "../Data/Feature.xlsx"
    TRAIN = "../Data/POS_train_val.csv"
    TEST = "../Data/POS_test.csv"

    # ---------- (1) 데이터 로드 ----------
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
    # 2-1. 운영시간
    df_all["operating_hours"] = df_all.apply(calculate_operating_hours, axis=1)

    # 2-2. 시험 window (optional)
    df_all["exam_before3"] = df_all["exam"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    df_all["exam_after3"] = df_all["exam"].shift(-1).rolling(3, min_periods=1).sum().fillna(0)

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
        # 현재 날 정보는 쓰지 않도록 shift(1)
        df_all[f"RollingMean{win}"] = roll.mean().shift(1)
        df_all[f"RollingStd{win}"] = roll.std(ddof=0).shift(1)

    # 2-6. weekday one-hot
    df_all = pd.get_dummies(df_all, columns=["weekday"], drop_first=True)

    print("\nAfter feature engineering:")
    print(df_all.head())

    # ---------- (3) Train / Test 분리 ----------
    # NaN 있는 행 제거 (Lag/Rolling 때문에 앞부분이 NaN)
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
    y_train = train["daily"].values
    X_val = val[feature_cols]
    y_val = val["daily"].values

    print("\nFeature columns:", len(feature_cols))
    print(feature_cols)

    # ---------- (5) XGBoost Grid Search (Val RMSE 기준) ----------
    from itertools import product

    # 튜닝할 하이퍼파라미터 후보들
    param_grid = {
        "max_depth":        [4, 5, 6],     # 트리 깊이
        "learning_rate":    [0.03, 0.05],  # 학습률 (작을수록 느리지만 안정적)
        "n_estimators":     [400, 800],    # 트리 개수
        "min_child_weight": [2, 4],        # 리프 분기 제한
        "subsample":        [0.7, 0.9],    # row sampling 비율
        "colsample_bytree": [0.7, 0.9],    # column sampling 비율
    }

    best_rmse = np.inf
    best_params = None

    print("\n===== Hyperparameter Search (based on Val RMSE, XGBoost) =====")

    for max_depth, lr, n_estimators, min_child_weight, subsample, colsample in product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["n_estimators"],
        param_grid["min_child_weight"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
    ):
        params = {
            "objective": "reg:squarederror",
            "max_depth": max_depth,
            "learning_rate": lr,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "random_state": 42,
            "n_jobs": -1,
        }

        reg_tmp = xgb.XGBRegressor(**params)

        reg_tmp.fit(X_train, y_train)
        val_pred_tmp = reg_tmp.predict(X_val)
        rmse_tmp = np.sqrt(mean_squared_error(y_val, val_pred_tmp))

        print(f"params={params} → RMSE={rmse_tmp:.4f}")

        if rmse_tmp < best_rmse:
            best_rmse = rmse_tmp
            best_params = params

    print("\n>>> Best Params (by Val RMSE):")
    print(best_params)
    print(f">>> Best Val RMSE: {best_rmse:,.4f}")

    # 🔹 최적 파라미터로 최종 모델 다시 학습
    reg = xgb.XGBRegressor(**best_params)
    reg.fit(X_train, y_train)

    # ---------- (6) Validation 평가 ----------
    val_pred = reg.predict(X_val)

    mae_val = mean_absolute_error(y_val, val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    smape_val = smape(y_val, val_pred)

    print("\n===== Validation Performance (with Lags & Rolling, tuned XGBoost) =====")
    print(f"MAE   : {mae_val:,.2f}")
    print(f"RMSE  : {rmse_val:,.2f}")
    print(f"SMAPE : {smape_val:.2f}%")

    print("\nFeature Importances (Top 30):")
    fi = pd.Series(reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(fi.head(30))

    # ---------- (7) Test 평가 (daily>0만) ----------
    df_test_eval = df_test[df_test["daily"] > 0].copy()
    if len(df_test_eval) == 0:
        print("\nNo test rows with daily>0 after lag/rolling/NaN filtering.")
        return

    X_test = df_test_eval[feature_cols]
    y_test = df_test_eval["daily"].values
    test_pred = reg.predict(X_test)

    mae_test = mean_absolute_error(y_test, test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
    smape_test = smape(y_test, test_pred)

    print("\n===== Test Performance (daily>0 only, tuned XGBoost) =====")
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
