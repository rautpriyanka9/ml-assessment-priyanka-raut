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

# B2. Data and EDA Strategy

## (a) Data Integration and Grain
To prepare the dataset for machine learning, the four disparate tables must be unified into a single "Feature Matrix."

* **Joining Logic:** * Use the **Transactions** table as the "Fact" table.
    * Perform **Left Joins** with the **Store Attributes** (key: `store_id`), **Calendar** (key: `transaction_date`), and **Promotion Details** (key: `promotion_id`) tables.
* **Dataset Grain:** * The final grain should be **One row per Store per Day**.
* **Required Aggregations:** * Since raw transactions are likely at the individual receipt level, we must aggregate to the daily level using `SUM(items_sold)` as our target variable.
    * If available, calculating `COUNT(transaction_id)` per day can serve as a proxy for **Footfall**, a high-value predictive feature.

## (b) Exploratory Data Analysis (EDA)
The following EDA steps are critical to understanding underlying patterns before model training:

| Analysis / Chart | What to Look For? | Influence on Modeling |
| :--- | :--- | :--- |
| **Time-Series Plot** | Seasonality (weekly/monthly) and long-term sales trends. | Helps decide on adding temporal features like `month` or `is_festival`. |
| **Box Plots (Sales vs Promo)** | Distribution and outliers in sales for each promotion type. | Identifies which promotions have high variance or minimal impact. |
| **Correlation Heatmap** | High correlation between features (e.g., store size and footfall). | Detects **Multicollinearity**; informs feature selection to remove redundant variables. |
| **Histograms (Target Distribution)** | Skewness in the `items_sold` distribution. | If data is heavily skewed, a **Log Transformation** may be applied to the target. |

[Image of a correlation heatmap and histograms for exploratory data analysis]

## (c) Handling Data Imbalance (No-Promotion Bias)
In this scenario, 80% of the data represents "business as usual" (no promotion). This imbalance can cause the model to over-fit to the baseline and under-predict the "lift" caused by marketing activities.

**Proposed Mitigation Steps:**
1.  **Stratified Temporal Splitting:** Ensure that both training and testing sets contain a representative ratio of "Promotion" vs "No-Promotion" days to validate the model's sensitivity to marketing changes.
2.  **Interaction Features:** Create mathematical interactions between `promotion_type` and `location_type`. This allows the model to learn that a promotion’s effectiveness is conditional on where the store is located.
3.  **Target Re-framing (Incremental Lift):** Instead of predicting absolute `items_sold`, the model can be trained to predict the **Lift** (Actual Sales minus the Store's Moving Average Baseline). This forces the model to focus specifically on the variance caused by the promotion.

[Image of interaction effects in machine learning]

# B3. Model Evaluation and Deployment

## (a) Evaluation Framework
To ensure the model performs reliably in a real-world retail environment, we must implement a rigorous temporal validation strategy.

* **Split Strategy:** Use a **Temporal (Time-Series) Split**. With three years of data, the first 30 months should be used for training/validation, and the final 6 months (the most recent data) should be used as the hold-out test set.
* **Why Random Split is Inappropriate:** Retail data is time-dependent. A random split would lead to **Look-ahead Bias**, where the model "sees" future patterns (like a December peak in year 3) to predict past sales (in year 1). This creates unrealistic performance metrics that will not hold up in production.
* **Key Metrics & Business Interpretation:**
    * **MAE (Mean Absolute Error):** Quantifies how many "items" the prediction is off by on average. (e.g., *"Our forecast is typically accurate within ±15 units"*).
    * **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily; critical for avoiding major stock-outs or excessive overstocking costs.
    * **R-Squared ($R^2$):** Measures what percentage of sales variance is explained by the features vs. random noise.

[Image of Time Series Walk-forward Validation]

## (b) Investigating and Communicating Model Decisions
If the model recommends different promotions for the same store in different months (e.g., Store 12: Loyalty in Dec vs. Flat Discount in Mar), we use **Explainable AI (XAI)** techniques.

1.  **Investigation (SHAP/LIME):** We analyze the local feature importance for both instances. 
    * **In December:** Features like `is_festival` and `month_12` likely have high positive weights. The model suggests Loyalty Points because holiday shoppers are already active and respond well to long-term reward incentives.
    * **In March:** Without holiday traffic, the model might see that `competition_density` is the primary driver. It suggests a "Flat Discount" to drive immediate price-sensitive traffic during a "dry" month.
2.  **Communication:** Present the marketing team with a **Waterfall Plot**. This visualizes the "plus/minus" impact of each feature on the final prediction, turning the "black box" into a clear narrative of cause and effect.

[Image of SHAP waterfall plot for model interpretability]

## (c) Deployment and Monitoring
The transition from a notebook to a production environment requires an automated inference pipeline.

1.  **Model Serialization:** Save the entire scikit-learn pipeline (preprocessing + regressor) using `joblib`. This ensures that scaling and encoding transformations are identical between training and production.
2.  **Inference Pipeline:**
    * **Data Preparation:** At the start of each month, a script pulls the latest store and calendar data.
    * **Scoring:** The model generates 5 predictions for every store (one for each promotion type).
    * **Decision:** The system outputs the `promotion_type` with the highest predicted `items_sold`.
3.  **Monitoring & Drift Detection:**
    * **Performance Decay:** We track actual vs. predicted sales monthly. If MAE exceeds a pre