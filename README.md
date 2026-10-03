# PROJECT 1: CUSTOMER CHURN PREDICTION
# WEEK 01
# Telco Customer Churn — Exploratory Data Analysis (EDA)

**Project Overview**
This project performs an end-to-end Exploratory Data Analysis (EDA) on the Telco Customer Churn dataset to identify key drivers of customer attrition. By analyzing demographic details, subscribed services, account information, and billing metrics, this analysis provides actionable insights to help optimize customer retention strategies and minimize churn.

---

# Key Insights & Findings

**Overall Attrition Rate:** The baseline churn rate across the dataset stands at 26.54%, indicating a significant portion of customers leaving the service.

**Tenure Impact:** Customer tenure is strongly inversely correlated with churn. The highest risk of churn occurs within the first 1-12 months of subscription.

**Contract Type Dynamics:** Customers on month-to-month contracts exhibit drastically higher churn rates compared to those on one-year or two-year contracts.

**Payment Methods & Billing:** Customers utilizing Electronic Checks show a disproportionately higher churn rate compared to automated payment methods (Bank Transfer, Credit Card).

**Internet & Add-On Services:** Fiber optic subscribers experience higher churn rates relative to DSL users, primarily driven by higher monthly costs and lack of attached tech support or online security add-ons.

----

# Dataset Architecture & Preprocessing

**Total Records:** 7,043 rows and 21 feature columns.

**Data Type Handling:** Converted TotalCharges from string/object to numeric, addressing missing white-space entries via median imputation.

---

# Feature Categories:

**Demographics:** gender, SeniorCitizen, Partner, Dependents

**Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

**Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

**Target Variable:** Churn (Yes / No)

---

# WEEK 02
# Telco Customer Churn Prediction & Analysis (Building-ML-Models)
A comprehensive machine learning pipeline and exploratory analysis predicting customer churn using the Telco Customer Churn dataset. This repository contains complete data preprocessing, baseline modeling, logistic regression, decision tree, and random forest evaluations, odds-ratio interpretations, confusion matrix analysis, and ROC-AUC performance metrics.

---

# 📊 Project Overview & Workflow
The analysis is structured into six main components:

**Preprocessing, Split, and Baseline:**

Handled missing values in TotalCharges (filling 11 blank rows with the median), one-hot encoded categorical features (resulting in 30 total features), performed an 80/20 stratified train-test split, and established a baseline model using a Dummy Classifier.   

**Logistic Regression & Odds Ratios:** 

Trained a scaled logistic regression pipeline (StandardScaler + LogisticRegression), evaluated coefficients, and calculated odds ratios to determine key risk and protective customer churn factors.   

**Decision Tree Classifier:**

Trained a non-linear decision tree model to capture complex, hierarchical feature interactions and evaluated its splitting rules and feature importances.

**Random Forest Classifier:** 

Built an ensemble random forest model to improve generalization, reduce overfitting, and compare overall predictive performance against linear and single-tree models.

**Confusion Matrix & Metrics:**

Computed evaluation metrics (Precision, Recall, F1-Score) both manually and via Scikit-Learn to analyze classification performance.   

**ROC-AUC & Threshold Analysis:** 

Generated the ROC curve and calculated the Area Under the Curve (AUC) to assess model discrimination capability.   

---

# 🔑 Key Results & Numerical Values

**Dataset Dimensions:** 

7,043 rows and 21 original columns, expanding to 30 features after encoding.   

**Baseline Accuracy (Dummy Classifier):** 

73.46% (0.735), which catches 0 churners and establishes the minimum performance threshold to beat.

**Logistic Regression Performance:**

            1. Overall Accuracy: 80.7% (0.807)   
            2. Precision (Churn = Yes): 0.658
            3. Recall (Churn = Yes): 0.567
            4. F1-Score (Churn = Yes): 0.609
            
**Top Risk Factors (Highest Odds Ratios):**

            1. InternetService_Fiber optic: 2.179 (More than doubles the odds of a customer churning)   
            2. TotalCharges: 1.644
            3. StreamingMovies_Yes: 1.295
            
**Top Protective Factors (Lowest Odds Ratios):**

            1. Tenure: 0.295
            2. MonthlyCharges: 0.398
            3. Contract_Two year: 0.555

**Confusion Matrix Breakdown:**

            1. True Negatives (TN): 925
            2. False Positives (FP): 110
            3. False Negatives (FN): 162 (actual churners missed by the model)   
            4. True Positives (TP): 212

**Decision Tree Performance:**

            1. Overall Accuracy: 78.7%
            2. Precision (Churn = Yes): 0.62
            3. Recall (Churn = Yes): 0.51
            4. F1-Score (Churn = Yes): 0.56

  
**Random Forest Performance:**

            1. Overall Accuracy: 80.7% 
            2. Precision (Churn = Yes): 0.66
            3. Recall (Churn = Yes): 0.54
            4. F1-Score (Churn = Yes): 0.59
            5. Test AUC: 0.852 
  <img width="1024" height="579" alt="1" src="https://github.com/user-attachments/assets/cdba2f34-fe1c-4054-8330-373d4ce611da" />


---

# 📈 Visualizations & Graphs

1. Confusion Matrix:

The confusion matrix below illustrates the predictive breakdown of customers who stayed versus those who churned, highlighting the model's performance in catching actual churners.

<img width="485" height="358" alt="WhatsApp Image 2026-09-25 at 4 24 54 PM" src="https://github.com/user-attachments/assets/2045dec4-0ff9-4994-b2f4-a485a0ab8ad5" />

2. ROC Curve:

The Receiver Operating Characteristic (ROC) curve evaluates the trade-off between the True Positive Rate (Recall) and False Positive Rate across varying probability thresholds.

<img width="481" height="375" alt="WhatsApp Image 2026-09-25 at 4 25 21 PM" src="https://github.com/user-attachments/assets/f197da6c-5b9f-4b76-a424-daebca40a2e8" />

3. Decision Tree:
   
A regularized decision tree with a restricted depth balances training and test performance to prevent overfitting while mapping out key churn split rules like tenure and fiber optic service.

<img width="476" height="368" alt="WhatsApp Image 2026-09-25 at 4 33 31 PM" src="https://github.com/user-attachments/assets/f7e719ea-d839-4d26-a6ce-f5ee4b3413be" />

4. Random Forest:
  
The Random Forest ensemble leverages bagging to achieve a strong test accuracy of $80.7\%$ and a test AUC of $0.852$, outperforming individual decision trees.

<img width="987" height="401" alt="WhatsApp Image 2026-09-25 at 4 53 33 PM" src="https://github.com/user-attachments/assets/9e9ace76-b499-44e2-bae0-514e81d92f53" />



---


# WEEK 03
# Telco Customer Churn - Model Optimization and Unsupervised Learning

An end-to-end Machine Learning case study evaluating validation split noise, cross-validation stability, hyperparameter tuning, and model comparison on the Telco Customer Churn dataset.


## 📌 Project Overview & Key Findings
This demonstrates the critical importance of proper cross-validation over single train-test splits when evaluating predictive models. Using the Telco Customer Churn dataset, this study shows how random split noise can easily mislead model selection and highlights the trade-offs between evaluation metrics (ROC-AUC vs. Recall).

---
### Key Insights
1. **Split-to-split accuracy range across 20 seeds:** `0.783` to `0.817` (Mean: `0.803`, SD: `0.010`).
2. **5-fold CV AUC:**
   * **Logistic Regression:** `0.846 ± 0.013`
   * **Random Forest:** `0.844 ± 0.011`
   * **XGBoost:** `0.841 ± 0.012`
3. **Tuning Efficiency:**
   * **Best RF Params:** `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}`
   * **Grid vs. Random Search Time:** Grid Search took `142.5s` vs. Random Search in `28.2s` (Random Search achieved identical best CV AUC while running **~5x faster**).
4. **Test AUC of final model (used once):** `0.848` (Evaluated on held-out 20% test set).
5. **Customer segments ($k = 3$):**
   * **Segment 0 (Low-Tenure Month-to-Month):** `47.2%` churn
   * **Segment 1 (Long-Term Two-Year Contract):** `2.8%` churn
   * **Segment 2 (Mid-Tenure One-Year Contract):** `11.5%` churn
6. **PCA Feature Compression:** **`11` of 30 components** explain $90\%$ of the total variance ($90.4\%$).
7. **Biggest lesson:** *A single train-test split introduces up to 3.4% noise variance, making stratified cross-validation essential to prevent misleading model selection.*

---

## 🛠️ Pipeline Architecture

* **Part 1: Test Set Lock & Split Noise Measurement**
  * Locks away a 20% holdout test set (`X_test`, `y_test`) to prevent data leakage.
  * Evaluates Logistic Regression across 20 different random train/validation splits.
  * Calculates empirical vs. theoretical standard error to quantify validation noise.
* **Part 2: 5-Fold Stratified Cross-Validation**
  * Evaluates Logistic Regression and Random Forest using 5-Fold Stratified CV.
  * Tracks `ROC-AUC`, `Recall`, and `F1-Score` to analyze trade-offs.
* **Part 3: Hyperparameter Tuning**
  * Computes and plots validation curves for Logistic Regression's regularization parameter ($C$) using `validation_curve`.

---

## 📊 Results & Visualizations

### 1. Split Noise Analysis (20 Random Seeds)
* **Accuracy Range:** $0.780 - 0.828$
* **Standard Deviation:** $0.0104$
* **Theoretical SE:** $0.0107 \rightarrow 95\%\text{ CI } \pm 0.021$

| Split Noise Visualization |
| :---: |
|<img width="530" height="369" alt="WhatsApp Image 2026-10-03 at 1 27 21 PM" src="https://github.com/user-attachments/assets/ff066c99-4d5b-450c-9657-bfc639d778da" />|

---

### 2. 5-Fold Cross-Validation Model Comparison

| Model | ROC-AUC | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **0.846 ± 0.013** | **0.545 ± 0.042** | **0.594 ± 0.030** |
| **Random Forest** | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |
|             |          |         |        |

---

### 3. Logistic Regression Hyperparameter Tuning ($C$)

| Validation Curve ($C$ Parameter) |
| :---: |
| <img width="537" height="371" alt="image (2)" src="https://github.com/user-attachments/assets/01942b15-ba3d-466a-b5a8-a1a644f4d1df" /> |

