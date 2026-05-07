# 📈 Retail Demand Forecasting Dashboard

A machine learning-based retail analytics system that predicts weekly sales for different store departments and provides business insights such as promotion impact, demand trends, and inventory suggestions.

---

## 🚀 Project Overview

This project predicts **weekly sales** for a given **store and department** using historical sales data, economic indicators, and promotional features.

It also provides:

* 📊 Sales forecasting
* 🎯 Promotion impact analysis
* 📦 Inventory recommendations
* 📈 Business insights dashboard

---

## 🧠 Problem Statement

Retail businesses need accurate demand forecasting to:

* Avoid overstocking or stockouts
* Plan promotions effectively
* Improve revenue and profitability

This project solves that using machine learning.

---

## 🛠️ Tech Stack

* Python
* Streamlit (Dashboard UI)
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Plotly

---

## 📂 Project Structure

```
retail-demand-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── model.pkl
│
├── notebooks/
│
├── app.py
├── train_upgrade.py
├── benchmark_model.py
│
├── README.md
├── requirements.txt
```

---

## 📊 Features

### 🔹 Forecasting

* Predict weekly sales for any store and department
* Compare with last year same week

### 🔹 Promotion Intelligence

* Analyze impact of different promotions
* Suggest best promotion strategy

### 🔹 Insights Dashboard

* Store type classification
* Department performance ranking
* Gap analysis vs top-performing department

### 🔹 Spillover Effect

* Identifies how promotions in one department affect others
* Example: Grocery promotions increasing Pharmacy sales

### 🔹 Economic Awareness

* Incorporates CPI, fuel price, unemployment
* Adjusts predictions based on economic conditions

---

## 📈 Model Used

### Final Model: **XGBoost Regressor**

#### Performance:

| Metric | Value   |
| ------ | ------- |
| MAE    | 1175.4  |
| RMSE   | 2428.66 |
| R²     | 0.9878  |

---

## ⚖️ Model Comparison

| Model         |        MAE |       RMSE |       R² |
| ------------- | ---------: | ---------: | -------: |
| Decision Tree |       High |       High |    Lower |
| Random Forest |     Medium |     Medium |     Good |
| XGBoost       | **Lowest** | **Lowest** | **Best** |

✅ XGBoost was selected due to highest accuracy and stability.

---

## 🧮 Key Features Used

* Lag features (Lag_1, Lag_2, Lag_4, Lag_13, Lag_52)
* Rolling averages (Rolling_Mean_4, Rolling_Mean_8)
* Seasonal encoding (Week_sin, Week_cos)
* Promotions (MarkDown1–5, Total_Markdown, Promo_Flag)
* Economic indicators (CPI, Fuel Price, Unemployment)

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the dashboard

```
streamlit run app.py
```

---

## 📌 Example Use Case

* Select **Store** and **Department**
* Choose a future date
* Apply promotions
* View:

  * Predicted sales
  * Profit estimate
  * Recommendation
  * Sales trend graph

---

## 💡 Business Value

* Helps inventory managers optimize stock
* Helps marketing teams design better promotions
* Reduces losses due to poor forecasting
* Improves overall store performance

---

## ⚠️ Limitations

* Predictions depend on historical patterns
* External sudden events (e.g., lockdowns) are not captured
* Promotion impact is estimated, not causal

---

## 🔮 Future Improvements

* Real-time data integration
* Deep learning models (LSTM)
* Multi-store comparison dashboard
* Inventory optimization engine
* Automated report generation

---



---

## ⭐ Conclusion

This project combines **machine learning + business logic** to create a practical retail forecasting tool that goes beyond simple predictions and provides actionable insights.
