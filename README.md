# 🛒 SmartCart AI — Retail Supply Chain & Inventory Optimizer

An AI-powered demand forecasting and inventory optimization platform built for retail supply chain analytics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartcartai.streamlit.app)

## 🚀 Features

- **Demand Forecasting** — Random Forest ML model predicts weekly product demand
- **Inventory Optimization** — EOQ (Economic Order Quantity) and Reorder Point calculations
- **Interactive Dashboard** — Real-time KPIs, revenue trends, and category breakdowns
- **Sales Trend Analysis** — Multi-product comparison across time

## 🛠 Tech Stack

- **Python** · **Scikit-Learn** · **Streamlit** · **Pandas** · **NumPy** · **Matplotlib**

## ⚙️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/SmartCartAI.git
cd SmartCartAI
pip install -r requirements.txt
streamlit run src/app.py
```

## 📁 Project Structure

```
SmartCartAI/
├── src/
│   ├── app.py          # Streamlit dashboard (main entry point)
│   ├── forecast.py     # Random Forest forecasting model
│   ├── inventory.py    # EOQ & reorder point engine
│   └── config.py       # App configuration
├── data/
│   └── retail_sales.csv
├── tests/
│   └── test_forecast.py
└── requirements.txt
```

## 🧠 ML Model

The forecasting model uses **Random Forest Regressor** with features:
- Week of year, Month, Quarter (temporal patterns)
- Product encoding (per-SKU demand behavior)

Outputs weekly unit forecasts for any product, 2–12 weeks ahead.

## 📊 Inventory Logic

- **Reorder Point** = (Avg Demand × Lead Time) + Safety Stock  
- **EOQ** = √(2DS/H) — classic square-root formula balancing order & holding costs
