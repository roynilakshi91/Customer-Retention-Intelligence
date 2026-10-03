# Customer Retention Intelligence

Predict customer churn risk and explore the factors influencing each prediction.

## Live app

[Open the Customer Retention Intelligence app](https://roynilakshi91-customer-retention-intelligence-app-1-n5saof.streamlit.app/)

# Customer Retention Intelligence

An end-to-end machine learning project that identifies customers at risk of churn, explains the key drivers behind each prediction, and translates model outputs into actionable retention strategies and business impact.

## Workflow

```text
EDA
 ↓
Feature Engineering
 ↓
Churn Prediction
 ↓
SHAP Explainability
 ↓
Revenue at Risk
 ↓
Retention Targeting
 ↓
Causal Inference + A/B Testing
 ↓
CUPED / Difference-in-Differences
 ↓
Incremental Revenue
 ↓
ROI Analysis
 ↓
Streamlit Deployment



## Project Overview

Customer churn is not only a prediction problem — it is a business problem.

This project builds a complete customer retention workflow:

**Explore → Engineer Features → Predict Risk → Explain Predictions → Identify Retention Opportunities → Measure Impact**

The project uses the Telco Customer Churn dataset containing **7,043 customers** and combines machine learning, explainable AI, and business analysis.

---

## Objectives

- Identify customers with a high probability of churn
- Understand the factors driving individual churn predictions
- Segment customers by churn risk
- Quantify potential revenue at risk
- Develop targeted retention strategies
- Evaluate a simulated retention intervention through A/B testing
- Estimate the potential financial impact of retention actions
- Deploy the model through an interactive Streamlit application

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit
- Joblib
---

## Project Workflow

### 1. Exploratory Data Analysis

Analyzed customer demographics, services, contracts, tenure, billing, and churn patterns.

Key findings include:

- Overall churn rate: **26.5%**
- Month-to-month customers show substantially higher churn than customers on longer-term contracts.
- Customer tenure and service adoption are important factors in churn behavior.

---

### 2. Feature Engineering

Created additional features to improve the model and support business interpretation:

- `charge_ratio`
- `avg_monthly_charge`
- `service_score`
- `tenure_group`

The engineered `charge_ratio` feature became one of the strongest SHAP drivers in the analysis. 

---

### 3. Machine Learning Model

Multiple modeling approaches were evaluated, with **XGBoost** selected as the final model.

Model evaluation:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8637** |
| Accuracy | **0.81** |
| Churn Precision | **0.69** |
| Churn Recall | **0.53** |
| Churn F1 | **0.60** |

The model was evaluated using a held-out dataset. 

---

### 4. SHAP Explainability

SHAP was used to understand why the model assigns a high churn probability to individual customers.

Top churn drivers identified through SHAP include:

1. Month-to-month contract
2. Charge ratio
3. No online security
4. Monthly charges
5. Fiber optic internet

SHAP explanations allow the application to move beyond:

> "This customer is high risk."

and answer:

> "Why is this customer high risk?"

---

### 5. Risk Segmentation

Customers are grouped into three risk tiers:

| Risk Tier | Churn Probability |
|---|---:|
| Low Risk | 0–30% |
| Medium Risk | 30–60% |
| High Risk | 60–100% |

The analysis identified:

- **4,398** Low Risk customers
- **1,728** Medium Risk customers
- **917** High Risk customers

---

### 6. Revenue at Risk

The model output is translated into financial impact.

The analysis estimated:

- Monthly revenue at risk: **$112,735**
- 24-month LTV at risk: **$2.71M**
- Estimated revenue recovered if 20% of predicted churners are retained: **$541K**

These figures are based on the project's stated assumptions and should be interpreted as estimates rather than realized business results. 

---

### 7. Retention Strategy

The model is used to identify targeted retention opportunities.

Examples include:

| Risk Signal | Potential Intervention |
|---|---|
| Month-to-month contract | Contract upgrade offer |
| High charge ratio | Personalized loyalty incentive |
| No online security | Security add-on trial |
| Fiber optic + limited add-ons | Proactive customer outreach |
| Electronic check | Auto-pay migration |

The goal is to connect **model predictions → customer action → measurable business impact**.

---

### 8. A/B Testing & ROI

A retention intervention is simulated for high-risk month-to-month customers.

**Control group**
- No special retention offer

**Treatment group**
- Receives a retention incentive

The analysis compares churn rates between the two groups using a two-proportion statistical test.

> **Note:** The A/B test is a simulation because the original dataset does not contain randomized treatment and control groups. The results should therefore not be interpreted as measured causal business impact.

ROI is then estimated by comparing the potential revenue retained against the intervention cost.

---

## Streamlit Application

The deployed Streamlit application provides a simple interface where users can:

1. Enter customer information
2. Generate a churn probability
3. View the customer's risk level
4. See the key factors driving the prediction
5. Understand the potential retention opportunity

