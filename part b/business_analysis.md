# Fashion Retail Promotion Effectiveness Analysis

## B1. Problem Formulation

### (a) Machine Learning Problem Definition
* **Target Variable:** `items_sold` 
    * *Description:* A continuous numerical value representing the total sales volume (quantity).
* **Candidate Input Features:**
    * **Store Characteristics:** `store_id`, `store_size`, `location_type`, `competition_density`.
    * **Marketing/Temporal Inputs:** `promotion_type`, `transaction_date` (extracted as year, month, and day of week), `is_weekend`, `is_festival`.
* **ML Problem Type:** **Regression**
    * **Justification:** The objective is to predict a specific, precise quantity (number of items) rather than a discrete category or class. Since the target variable is continuous, a regression framework is required to model the relationship between store/promotion features and the resulting sales volume.

### (b) Target Variable Selection: Sales Volume vs. Revenue
* **Reliability of `items_sold`:** Total sales revenue can be heavily skewed by the nature of the promotion. For example, a "Flat 50% Discount" might significantly increase the number of items sold while resulting in lower total revenue compared to a "Free Gift" promotion. `items_sold` is a more reliable metric because it directly measures **consumer demand and promotion resonance** without the noise of price-point variations.
* **Broader Principle:** This illustrates the principle of **Objective Proximal Mapping**. In real-world ML, the target variable should be as close as possible to the specific behavior you are trying to influence. By selecting volume over revenue, we isolate the "promotion effectiveness" signal from the "pricing strategy" noise.

### (c) Proposed Alternative Modeling Strategy
* **Strategy:** **Segmented Modeling (or Stratified Regression)**
* **Description:** Instead of a single global model, develop sub-models for different store segments, specifically based on `location_type` (Urban, Semi-Urban, and Rural).
* **Justification:** Customer behavior often varies drastically by geography. For example, Urban stores might see higher lifts from "Loyalty Points," while Rural stores might respond more strongly to "BOGO" offers. A single global model tends to "average out" these critical nuances. Segmented modeling allows the algorithm to learn unique feature weights for each environment, leading to more accurate, localized promotion recommendations.

---

## Q3. Technical Implementation Summary

### 1. Feature Engineering
From the `transaction_date`, the following features were derived to capture seasonality and monthly cycles:
* `year`, `month`, `day_of_week`
* `is_month_end`: A binary flag set to 1 if the day of the month is $\ge 25$, capturing end-of-month shopping surges.

### 2. Pipeline Architecture
A scikit-learn `Pipeline` was utilized to ensure a reproducible and leak-free workflow:
* **Categorical Encoding:** `OneHotEncoder` applied to `promotion_type`, `location_type`, and `store_size`.
* **Numerical Scaling:** `StandardScaler` applied to all numerical features.
* **Temporal Split:** Data was split using an 80/20 chronological split (rather than random) to prevent **temporal data leakage** and ensure the model is tested on "future" data.

### 3. Model Performance & Insights
Based on the implementation:
* **Linear Regression** and **Random Forest Regressor** were compared.
* **Key Driver:** `is_festival` and `store_size` were identified as the most influential features in predicting sales volume.