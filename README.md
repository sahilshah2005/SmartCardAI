# SmartCart AI

AI-Powered Demand Forecasting and Inventory Optimization Platform for Retail Supply Chain Analytics

## Overview

SmartCart AI is a machine learning-based retail analytics platform designed to help businesses forecast product demand and optimize inventory management. The system analyzes historical sales data to generate demand predictions and inventory recommendations, enabling retailers to reduce stock-outs, minimize excess inventory, and improve supply chain efficiency.

The project demonstrates the application of data analytics and machine learning techniques in solving real-world retail inventory challenges.

## Problem Statement

Retail businesses often face challenges in maintaining optimal inventory levels. Insufficient stock can result in lost sales and dissatisfied customers, while excessive inventory increases storage costs and ties up working capital.

SmartCart AI addresses this problem by:

* Forecasting future product demand
* Identifying inventory risks
* Generating stock replenishment recommendations
* Supporting data-driven inventory planning

## Objectives

* Analyze historical retail sales data
* Forecast future product demand using machine learning
* Improve inventory decision-making
* Provide actionable business insights
* Demonstrate practical use of predictive analytics in retail

## Features

### Demand Forecasting

Predict future sales trends based on historical sales records.

### Inventory Optimization

Recommend inventory adjustments based on forecasted demand.

### Retail Analytics

Generate insights into product performance and sales trends.

### Risk Identification

Identify potential stock shortages and overstock situations.

### Dashboard Visualization

Present analytical results through an interactive dashboard.


## Project Structure

```text
SmartCartAI/
│
├── data/
│   └── retail_sales.csv
│
├── docs/
│   └── ARCHITECTURE.md
│
├── src/
│   ├── main.py
│   ├── app.py
│   ├── forecast.py
│   ├── inventory.py
│   └── config.py
│
├── tests/
│   └── test_forecast.py
│
├── requirements.txt
└── README.md
```

## Technology Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Streamlit

### Development Tools

* Git
* GitHub
* VS Code

## Dataset

The project uses a retail sales dataset containing:

* Day
* Sales Volume
* Product Category

The dataset is stored in:

```text
data/retail_sales.csv
```

## System Workflow

1. Load retail sales data
2. Perform data preprocessing
3. Train forecasting model
4. Generate demand predictions
5. Calculate inventory recommendations
6. Display results through dashboard

## How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/<your-username>/SmartCartAI.git
```

### Step 2: Navigate to Project Directory

```bash
cd SmartCartAI
```

### Step 3: Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Main Application

```bash
python main.py
```

### Step 6: Launch Dashboard

```bash
streamlit run app.py
```

### Step 7: Open Browser

Visit:

```text
http://localhost:8501
```

## Expected Output

* Demand Forecast Report
* Inventory Recommendations
* Sales Trend Analysis
* Dashboard Visualizations

## Future Enhancements

* Multi-warehouse inventory optimization
* Deep learning forecasting models
* Real-time sales integration
* Product recommendation engine
* Cloud deployment
* Advanced business intelligence dashboards


## Applications

* E-commerce platforms
* Retail chains
* Warehouse management
* Supply chain planning
* Inventory management systems


## Author

Sahil Shah

Final Year Computer Engineering Student

Mumbai University

```
```
