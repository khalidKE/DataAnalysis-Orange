#  Customer Satisfaction AI Prediction Engine

An elite, high-precision machine learning system designed to predict customer satisfaction and identify churn risks with surgical precision. This project leverages advanced feature engineering and a heavy-duty stacking ensemble to break through traditional accuracy barriers in CX analytics.

## 🌟 Overview

This engine transforms raw customer support tickets into actionable intelligence. By focusing on the most critical 20% of "Extremely Unhappy" customers (Rating 1), the system enables proactive intervention before churn occurs.

### Key Performance Metric
- **Current Accuracy:** 80.29% (Optimizing for 85%+)
- **Target:** Identifying Churn Risk with High Precision

---

## 🛠️ Tech Stack

- **Linguistics & Text:** `TfidfVectorizer` for sentiment proxying.
- **Data Science:** `Pandas`, `NumPy`, `RobustScaler`, `LabelEncoder`.
- **Machine Learning:** 
  - `StackingClassifier` (Elite Ensemble)
  - `XGBoost` (Extreme Gradient Boosting)
  - `HistGradientBoosting`
  - `LogisticRegression` (Final Estimator)
- **Class Balancing:** `SMOTE` (Synthetic Minority Over-sampling Technique).
- **Explainability:** `SHAP` (Shapley Additive Explanations).
- **Visualizations:** `Plotly`, `Matplotlib`.

---

## 🔬 Core Innovations

### 1. Pressure Non-linear Scaling
We developed a custom **'Pressure_Factor'** that weights wait time against ticket priority non-linearly. This captures the "boiling point" where customer frustration exponentially increases with every hour of delay.

### 2. Resolution Status Intelligence
Beyond simple status checks, the model tracks the presence of a solution. Tickets without a recorded 'Resolution' were found to be **5x more likely** to result in a 1-star rating.

### 3. Demographic Sensitivity
The model revealed that wait time is **2x more damaging** for younger customers compared to senior demographics, allowing for age-sensitive SLA prioritization.

---

## 📊 Visualizations

The system generates high-impact visualizations to explain model behavior:

![Customer Satisfaction Distribution](Customer%20satisfaction.png)
*Distribution of satisfaction ratings across the dataset.*

![Feature Importance](Features%20Importance.png)
*SHAP-based feature importance ranking identifying 'Res_Hours' as the primary driver.*

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Installation:
```bash
pip install -r requirements.txt
```

### Execution
Run the main prediction engine:
```bash
python main_project.py
```

### Outputs
- `complaints_cleaned_for_dashboard.csv`: Enriched dataset for BI tools.
- `feature_importance.csv`: Raw SHAP values for further analysis.
- `ai_insights_summary.md`: Auto-generated summary of the latest run's performance.

---

## 🏆 Project Achievements

- **Precision Targeting:** Successfully identifies the most critical customer segments.
- **Advanced Ensemble:** Implementation of a multi-stage stacking architecture for maximum robustness.
- **Explainable AI:** Integrated SHAP values to transform "black box" predictions into clear business roadmap items.



