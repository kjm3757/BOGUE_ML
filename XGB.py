import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# =========================
# 0. SMAPE 함수 정의
# =========================
def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100


# =========================
# 1. 데이터 불러오기 (Train/Val + Test)
# =========================
feature_path = "Feature.xlsx"
pos_trainval_path = "POS_train_val.csv"
pos_test_path = "POS_test.csv"

feat = pd.read_excel(feature_path)
pos_tv = pd.read_csv(pos_trainval_path)
pos_test = pd.read_csv(pos_test_path)

# 숫자형 컬럼 전처리
for col in ["daily", "AOV"]:
    pos_tv[col] = (
        pos_tv[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    pos_test[col] = (
        pos_test[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

pos_tv["date"] = pd.to_datetime(pos_tv["date"])
pos_test["date"] = pd.to_datetime(pos_test["date"])
feat["date"] = pd.to_datetime(feat["date"])

# =========================
# 2. Feature 머지 (Train/Val, Test 각각)
# =========================
df_tv_raw = pd.merge(pos_tv, feat, on="date", how="inner")
df_test_raw = pd.merge(pos_test, feat, on="date", how="left")

print("머지 후 Train/Val 데이터 크기:", df_tv_raw.shape)
print("머지 후 Test 데이터 크기:", df_test_raw.shape)

# =========================
# 3. 매출 0원인 날 제거 (Train/Val + Test 모두)
# =========================
df_tv = df_tv_raw[df_tv_raw["daily"] > 0].copy()
df_test = df_test_raw[df_test_raw["daily"] > 0].copy()

df_tv["set"] = "trainval"
df_test["set"] = "test"

print("0원 제거 후 Train/Val 데이터 크기:", df_tv.shape)
print("0원 제거 후 Test 데이터 크기:", df_test.shape)

# =========================
# 4. Train/Val + Test 합쳐서 Lag/Rolling 한 번에 계산
# =========================
df_all = pd.concat([df_tv, df_test], ignore_index=True)
df_all = df_all.sort_values("date").reset_index(drop=True)

# ----- Lag Feature -----
lag_list = [1, 2, 3, 7, 14, 28]
for l in lag_list:
    df_all[f"Lag{l}"] = df_all["daily"].shift(l)

# ----- Rolling Mean / Std (미래 누설 방지: shift(1) 후 rolling) -----
window_list = [7, 14, 28]
for w in window_list:
    df_all[f"RollingMean{w}"] = df_all["daily"].shift(1).rolling(window=w).mean()
    df_all[f"RollingStd{w}"] = df_all["daily"].shift(1).rolling(window=w).std()

# Lag/Rolling으로 생긴 NaN 제거
df_all_model = df_all.dropna().copy()
print("Lag/Rolling 적용 후 사용 가능 데이터 크기:", df_all_model.shape)

# =========================
# 5. weekday one-hot 인코딩
# =========================
df_all_model = pd.get_dummies(df_all_model, columns=["weekday"], drop_first=True)

# =========================
# 6. Feature / Target 설정
# =========================
target_col = "daily"
drop_cols = ["date", "daily", "num", "AOV", "set"]

feature_cols = [c for c in df_all_model.columns if c not in drop_cols]

print("\n사용 Feature 컬럼 목록:")
print(feature_cols)

# Train/Val, Test 다시 분리
df_tv2 = df_all_model[df_all_model["set"] == "trainval"].copy()
df_test2 = df_all_model[df_all_model["set"] == "test"].copy()

df_tv2 = df_tv2.sort_values("date")
df_test2 = df_test2.sort_values("date")

# =========================
# 7. Train / Val Split (Train/Val의 80:20)
# =========================
split_idx = int(len(df_tv2) * 0.8)

train = df_tv2.iloc[:split_idx]
val = df_tv2.iloc[split_idx:]

X_train = train[feature_cols]
y_train = train[target_col]
X_val = val[feature_cols]
y_val = val[target_col]

print("\nTrain shape:", X_train.shape, " Val shape:", X_val.shape)
print("Test shape:", df_test2[feature_cols].shape)

# =========================
# 8. XGBoost 모델 학습
# =========================
xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
)

xgb_reg.fit(X_train, y_train)

# =========================
# 9. Validation 성능 평가
# =========================
y_val_pred = xgb_reg.predict(X_val)

mae_val = mean_absolute_error(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
smape_val = smape(y_val.values, y_val_pred)

print("\n===== Validation 성능 (0원 제거 + Lag/Rolling, XGBoost) =====")
print(f"MAE   : {mae_val:,.2f}")
print(f"RMSE  : {rmse_val:,.2f}")
print(f"SMAPE : {smape_val:.2f}%")


# =========================
# 10. Test셋 예측 및 평가 (0원 제거된 날들만)
# =========================
X_test = df_test2[feature_cols]
y_test = df_test2[target_col]

y_test_pred = xgb_reg.predict(X_test)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
smape_test = smape(y_test.values, y_test_pred)

print("\n===== Test 성능 (0원 제거 + Lag/Rolling, XGBoost) =====")
print(f"MAE   : {mae_test:,.2f}")
print(f"RMSE  : {rmse_test:,.2f}")
print(f"SMAPE : {smape_test:.2f}%")


# =========================
# 11. Test 결과 테이블 확인
# =========================
result_test = pd.DataFrame({
    "date": df_test2["date"].values,
    "actual_daily": y_test.values,
    "pred_daily": y_test_pred.astype(int)
})

print("\n=== Test 예측 결과 상위 20개 ===")
print(result_test.head(20))
