# Telco Customer Churn Prediction: An End-to-End ML Pipeline

## Project Overview
Customer churn is one of the most critical metrics for any telecommunications business. Retaining an existing customer is significantly cheaper than acquiring a new one. This project builds a **Production-Ready Machine Learning Pipeline** to predict customer churn, identify at-risk customers, and uncover the key business drivers behind customer attrition.

## Key Engineering Decisions & MLOps Mindset
This project goes beyond simple model training by implementing best practices for production environments:

* **Robust Preprocessing Pipeline:** Utilized `scikit-learn`'s `Pipeline` and `ColumnTransformer` to ensure clean, reproducible code and eliminate data leakage.
* **Smart Imbalance Handling:** Instead of injecting synthetic noise with SMOTE, the class imbalance was handled natively inside the `XGBoost` algorithm using `scale_pos_weight`. This optimized memory usage and training speed while focusing the model on the minority class.
* **Feature Engineering:** Created new features like `TotalServices` to capture the depth of a customer's engagement with the company, giving the model stronger signals to learn from.
* **Avoiding Sparsity:** Replaced `OneHotEncoder` with `OrdinalEncoder` for Tree-based models. This prevented "importance fragmentation" and allowed XGBoost to correctly evaluate the full impact of categorical features without creating sparse matrices.
* **Hyperparameter Tuning:** Applied `RandomizedSearchCV` to fine-tune the XGBoost model, implementing proper regularization techniques (`subsample`, `max_depth`, `learning_rate`) to prevent overfitting on a medium-sized dataset.

## Model Performance
In churn prediction, optimizing for **Recall** (catching those who will actually leave) is far more important than raw Accuracy. 
After tuning, the **XGBoost** model achieved:
* **Recall (Churn Class):** `78%` (Successfully identifying ~8 out of 10 leaving customers).
* **F1-Score (Churn Class):** `0.63`
* **Overall Accuracy:** `76%`

## Key Business Insights 
Based on the **Feature Importance** and **Model Performance** analysis, here are the key takeaways for the retention team:

1. **The Contract Trap:** * **Insight:** `Contract Type` is the #1 predictor of churn (Importance: ~40%). Customers on month-to-month contracts are significantly more likely to leave.
   * **Action:** Launch an "Upgrade to Annual" campaign offering a 1-2 month discount for customers who switch from monthly to yearly contracts.

2. **Internet Service Quality:**
   * **Insight:** `Internet Service` type is the second most important factor. If Fiber-optic churn is high, it may indicate technical instability or price sensitivity in that segment.
   * **Action:** Audit the technical performance of Fiber-optic connections and compare pricing against competitors.

3. **"Sticky" Services (Retention Add-ons):**
   * **Insight:** `Online Security` and `Tech Support` are top drivers for retention. Customers without these features churn at a higher rate.
   * **Action:** Bundle Security and Tech Support for free or at a reduced price for "High Risk" monthly customers to increase their "switching cost."

4. **Strategic Model Selection (XGBoost vs. Others):**
   * **XGBoost** was chosen as the final model because it achieved a **78% Recall** for churners.
   * **Business Value:** While Logistic Regression had higher recall, it was too "noisy" (Low Precision). XGBoost provides the best balance, allowing the marketing team to target the *right* customers without wasting budget on loyal ones.
