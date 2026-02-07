#  Machine Learning & Data Analysis Projects

This repository is a collection of data analysis and machine learning projects covering regression, classification, exploratory data analysis (EDA), and real-world business use cases such as HR analytics, sales forecasting, customer satisfaction prediction, and loan data profiling.

Each project is organized in its own branch for clarity and modular development.

---

## 📂 Repository Structure (Branches Overview)

### 🔹 ML  
**Machine Learning with Scikit-Learn (Regression & Classification)**

This project demonstrates core ML workflows using Python and scikit-learn.

**Key Topics**
- Simple Linear Regression (synthetic data)
- Binary Classification (Breast Cancer Detection)

**Tech Stack**
- Python, NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

**Highlights**
- Feature scaling with `MinMaxScaler`
- Pipelines for clean preprocessing
- Logistic Regression accuracy ~96.5%
- Confusion matrix & visualization

---

### 🔹 MID-project  
**📊 Data Analysis & Sales Prediction Project**

An end-to-end data analytics pipeline that cleans, merges, analyzes, and models sales data, with Power BI integration.

**Features**
- Automated data cleaning & merging (Fact + Dimension tables)
- Sales prediction using Linear Regression
- Customer segmentation using Logistic Regression
- Power BI dashboard integration

**Outputs**
- Master_Cleaned_Sales_Data.csv
- ML performance reports
- Interactive Power BI dashboard (`.pbix`)

---

### 🔹 final  
**🤖 Customer Satisfaction AI Prediction Engine**

An advanced machine learning system designed to predict customer dissatisfaction and churn risk.

**Core Innovations**
- Stacking ensemble (XGBoost, HistGradientBoosting, Logistic Regression)
- Custom non-linear pressure feature engineering
- SMOTE class balancing
- SHAP explainability

**Performance**
- Current Accuracy: ~80.3%
- Focus on identifying critical 1-star customers

**Outputs**
- Enriched datasets for BI tools
- SHAP feature importance
- Auto-generated AI insights summary

---

### 🔹 HR  
**👥 HR Data Analysis & Machine Learning**

Predicts employee turnover and analyzes performance drivers using structured HR data.

**Models Used**
- Logistic Regression
- Random Forest
- Gradient Boosting

**Key Insights**
- Turnover prediction with imbalance handling
- Training impact on performance
- Department-level satisfaction analysis

---

### 🔹 DataPreprocessing-Loans-data  
**💰 Loan Data Analysis & Profiling**

Exploratory Data Analysis (EDA) and automated profiling of loan data.

**Key Components**
- Data cleaning & type conversion
- Descriptive statistics & correlation analysis
- Visualizations (boxplots, histograms, heatmaps)
- Automated profiling with `ydata-profiling`

**Output**
- `loan_data_report.html`

---

## 🛠 Technologies Used
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost, SHAP
- Power BI



Feel free to explore each branch for full implementation details.
