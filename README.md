# ml-assessment-priyanka-raut
# Multi-Domain Machine Learning Framework
**Healthcare Diagnostics | Customer Segmentation | Retail Sales Forecasting**

## 📌 Project Overview
This project demonstrates the application of Machine Learning across three distinct domains. It covers the end-to-end data science lifecycle, including data cleaning, class imbalance handling, dimensionality reduction, and the construction of reproducible scikit-learn pipelines.

---

## 📂 Repository Contents

* **`q1_supervised.ipynb`**: Heart disease classification using Random Forests.
* **`q2_unsupervised.ipynb`**: K-Means clustering and PCA for retail customer segmentation.
* **`q3_feature_engineering.ipynb`**: Scikit-learn regression pipeline for sales forecasting.
* **`q3_retail_promotions.csv`**: Daily transactional and promotional retail data.

---

## 🏥 Module 1: Supervised Learning (Heart Disease Prediction)
**Goal:** Classify patients based on the presence of heart disease using clinical markers.

### Key Implementation Details:
* **Missing Value Imputation:** Handled gaps in `resting_bp` and `cholesterol`.
* **Preprocessing:** Encoded categorical features (`chest_pain_type`, `st_slope`) using One-Hot Encoding.
* **Algorithm:** A **Random Forest Classifier** was optimized via `GridSearchCV`.
* **Performance:** Achieved an F1-score of ~0.78, indicating balanced precision and recall for both positive and negative cases.

[Image of a confusion matrix and classification report for a heart disease model]

---

## 🛍️ Module 2: Unsupervised Learning (Customer Personas)
**Goal:** Segment a fashion retailer’s customer base to enable targeted marketing.

### Technical Approach:
* **Feature Scaling:** Standardized demographic and spend data to ensure uniform distance calculations.
* **Dimensionality Reduction:** Applied **PCA** to reduce the feature set while maintaining maximum variance, allowing for clearer cluster boundaries.
* **Clustering:** Utilized **K-Means Clustering**. The optimal number of clusters was determined using the Elbow Method and Silhouette Analysis.
* **Result:** Identified distinct segments such as "High-Value Loyalists" and "Occasional Budget Shoppers."

[Image of PCA scatter plot showing distinct clusters of customer segments]

---

## 📈 Module 3: Retail Regression Pipeline & Strategy

### 1. Technical Pipeline (`q3_feature_engineering.ipynb`)
* **Date Engineering:** Derived `day_of_week`, `month`, and a binary `is_month_end` (day $\ge$ 25) to capture end-of-month spending surges.
* **Temporal Splitting:** Data was sorted by date, with the final 20% used as
