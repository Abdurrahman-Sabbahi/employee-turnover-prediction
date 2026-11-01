# Employee Turnover Prediction

> Predict employee turnover using Logistic Regression, Random Forest, and Gradient Boosting with SMOTE, K-Means clustering, and 5-fold cross-validation — plus data-driven retention strategy recommendations.

---

## Overview

This project is an end-to-end machine learning analysis of employee attrition for **Portobello Tech's HR department**. It covers the full ML pipeline: data quality checks, exploratory analysis, clustering of at-risk employees, class-imbalance handling, multi-model training and evaluation, and actionable retention strategy recommendations.

## Project Structure

```
├── employee_turnover_analysis_final.ipynb   # Main notebook
├── HR_comma_sep.csv                         # HR dataset
└── README.md
```

## Steps Covered

1. **Data Quality Checks** — missing values, duplicates, data types
2. **Exploratory Data Analysis (EDA)** — distributions, correlations, visualisations
3. **K-Means Clustering** — segment employees who left into behavioural clusters
4. **Class Imbalance Handling** — SMOTE oversampling on the minority (left) class
5. **Model Training & Evaluation** — 5-fold stratified cross-validation across three classifiers:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
6. **Best Model Selection** — ROC/AUC curves, confusion matrices, classification reports
7. **Retention Strategy Recommendations** — data-driven HR insights per cluster

## Dataset

`HR_comma_sep.csv` — anonymised HR records with features including satisfaction level, number of projects, average monthly hours, tenure, work accidents, promotions, department, and salary band.

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage

Open and run the notebook top-to-bottom:

```bash
jupyter notebook employee_turnover_analysis_final.ipynb
```

## Key Results

- Best classifier selected by ROC-AUC score via 5-fold cross-validation
- K-Means clusters reveal distinct attrition profiles (e.g. overworked high-performers vs. disengaged low-performers)
- Targeted retention strategies derived per cluster

## Certificate Context

Developed as the capstone ML project for the **Michigan Professional Certificate in AI/ML**.
