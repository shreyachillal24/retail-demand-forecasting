# src/model_benchmark.py

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = pd.read_csv("data/processed/train_features.csv")
df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------------------------------
# AUTO FEATURE SELECTION (SAFE)
# ---------------------------------------------------
feature_cols = [col for col in df.columns if col not in ["Weekly_Sales", "Date"]]

target_col = "Weekly_Sales"

X = df[feature_cols].copy()
y = df[target_col].copy()


# ---------------------------------------------------
# FIX DATA TYPES (CRITICAL)
# ---------------------------------------------------
cat_cols = ["Store", "Dept", "Type", "IsHoliday"]
cat_cols = [c for c in cat_cols if c in X.columns]

# Convert categorical to string
for col in cat_cols:
    X[col] = X[col].astype(str)

# Numeric columns = everything else
num_cols = [c for c in feature_cols if c not in cat_cols]


# ---------------------------------------------------
# TIME SPLIT
# ---------------------------------------------------
cutoff_date = pd.Timestamp("2012-07-01")

train_idx = df["Date"] < cutoff_date
test_idx  = df["Date"] >= cutoff_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, cat_cols),
    ("num", num_pipeline, num_cols)
])


# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}


# ---------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------
def evaluate_model(name, model):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)

    return {
        "Model": name,
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }


# ---------------------------------------------------
# RUN BENCHMARK
# ---------------------------------------------------
results = []

for name, model in models.items():
    print(f"Running {name}...")
    res = evaluate_model(name, model)
    results.append(res)

results_df = pd.DataFrame(results).sort_values(by="RMSE")

print("\n📊 Model Comparison Results:")
print(results_df.to_string(index=False))