import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "data/processed/train_features.csv"

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# cyclical week encoding
df["Week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
df["Week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)

# Promotion features
promo_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]

for c in promo_cols:
    if c not in df.columns:
        df[c] = 0

df[promo_cols] = df[promo_cols].fillna(0)

df["Total_Markdown"] = df[promo_cols].sum(axis=1)
df["Promo_Flag"] = (df["Total_Markdown"] > 0).astype(int)

# Group key
grp = ["Store", "Dept"]

# Lag features
df["Lag_1"] = df.groupby(grp)["Weekly_Sales"].shift(1)
df["Lag_2"] = df.groupby(grp)["Weekly_Sales"].shift(2)
df["Lag_4"] = df.groupby(grp)["Weekly_Sales"].shift(4)
df["Lag_13"] = df.groupby(grp)["Weekly_Sales"].shift(13)
df["Lag_52"] = df.groupby(grp)["Weekly_Sales"].shift(52)

# Same week last year
df["Same_Week_Last_Year"] = df["Lag_52"]

# Rolling means (use shifted values only)
df["Rolling_Mean_4"] = (
    df.groupby(grp)["Weekly_Sales"]
      .shift(1)
      .rolling(4)
      .mean()
)

df["Rolling_Mean_8"] = (
    df.groupby(grp)["Weekly_Sales"]
      .shift(1)
      .rolling(8)
      .mean()
)

# Fill missing lag values safely
lag_fill_cols = [
    "Lag_1","Lag_2","Lag_4","Lag_13","Lag_52",
    "Same_Week_Last_Year",
    "Rolling_Mean_4","Rolling_Mean_8"
]

for c in lag_fill_cols:
    df[c] = df[c].fillna(df["Weekly_Sales"].median())

# -----------------------------
# TRAIN / TEST SPLIT (TIME BASED)
# -----------------------------
cutoff_date = pd.Timestamp("2012-07-01")

train_df = df[df["Date"] < cutoff_date].copy()
test_df  = df[df["Date"] >= cutoff_date].copy()

# -----------------------------
# FEATURES
# -----------------------------
feature_cols = [
    "Store","Dept","Type",
    "Size","IsHoliday",
    "Temperature","Fuel_Price","CPI","Unemployment",

    "Year","Month","Week",
    "Week_sin","Week_cos",

    "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
    "Total_Markdown","Promo_Flag",

    "Lag_1","Lag_2","Lag_4","Lag_13","Lag_52",
    "Same_Week_Last_Year",
    "Rolling_Mean_4","Rolling_Mean_8"
]

X_train = train_df[feature_cols].copy()
y_train = train_df["Weekly_Sales"]

X_test = test_df[feature_cols].copy()
y_test = test_df["Weekly_Sales"]

# -----------------------------
# PREPROCESSING
# -----------------------------
cat_cols = ["Store", "Dept", "Type", "IsHoliday"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# force categorical columns to string
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# -----------------------------
# MODEL
# -----------------------------
USE_XGB = True

try:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

except:
    from sklearn.ensemble import HistGradientBoostingRegressor

    USE_XGB = False

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=8,
        random_state=42
    )

pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", model)
])

# -----------------------------
# TRAIN
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------

pred = pipe.predict(X_test)

# Main Metrics
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

# MAPE %
mask = y_test > 100   # ignore tiny / zero sales rows
mape = np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100
print("MAPE % :", round(mape,2))

# Baseline = Last Week Sales
baseline_pred = X_test["Lag_1"]
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("=" * 60)
print("MODEL:", "XGBoost" if USE_XGB else "HistGradientBoosting")
print("MAE           :", round(mae, 2))
print("RMSE          :", round(rmse, 2))
print("R2 Score      :", round(r2, 4))
print("MAPE %        :", round(mape, 2))
print("Baseline MAE  :", round(baseline_mae, 2))
print("=" * 60)

# Sample Predictions
compare = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": pred[:10],
    "Error": abs(y_test.values[:10] - pred[:10])
})

print("\nSample Predictions:")
print(compare)

# -----------------------------
# SAVE MODEL
# -----------------------------

import os
os.makedirs("model", exist_ok=True)
SAVE_PATH = "model/model.pkl"

joblib.dump(pipe, SAVE_PATH)

print("Saved model to:", SAVE_PATH)


import joblib

joblib.dump(feature_cols, "model/features.pkl")