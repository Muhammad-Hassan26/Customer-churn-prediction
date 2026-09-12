
# Telco Customer Churn — Exploratory Data Analysis (EDA)

**Project Overview**
This project performs an end-to-end Exploratory Data Analysis (EDA) on the Telco Customer Churn dataset to identify key drivers of customer attrition. By analyzing demographic details, subscribed services, account information, and billing metrics, this analysis provides actionable insights to help optimize customer retention strategies and minimize churn.

# Key Insights & Findings

**Overall Attrition Rate:** The baseline churn rate across the dataset stands at 26.54%, indicating a significant portion of customers leaving the service.

**Tenure Impact:** Customer tenure is strongly inversely correlated with churn. The highest risk of churn occurs within the first 1-12 months of subscription.

**Contract Type Dynamics:** Customers on month-to-month contracts exhibit drastically higher churn rates compared to those on one-year or two-year contracts.

**Payment Methods & Billing:** Customers utilizing Electronic Checks show a disproportionately higher churn rate compared to automated payment methods (Bank Transfer, Credit Card).

**Internet & Add-On Services:** Fiber optic subscribers experience higher churn rates relative to DSL users, primarily driven by higher monthly costs and lack of attached tech support or online security add-ons.

# Dataset Architecture & Preprocessing

**Total Records:** 7,043 rows and 21 feature columns.

**Data Type Handling:** Converted TotalCharges from string/object to numeric, addressing missing white-space entries via median imputation.

# Feature Categories:

**Demographics:** gender, SeniorCitizen, Partner, Dependents

**Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

**Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

**Target Variable:** Churn (Yes / No)
