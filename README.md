# MID: Data Analysis & Sales Prediction Project

## 📊 Project Overview
This project is a comprehensive data analysis and machine learning pipeline focused on sales data. It processes raw transactional and dimensional data, cleans and merges it into a master dataset, and performs advanced analytics including predictive modeling and customer segmentation.

The project also includes Power BI integration for interactive dashboards and automated reporting.

## 📂 Project Structure

```
MID/
│
├── clean_data.py                  # Core script to clean and merge raw CSV/Excel files
├── sales_prediction_model.py      # Linear Regression model to predict sales totals
├── customer_segmentation_model.py # Logistic Regression model to classify 'High Value' customers
├── generate_profiling.py          # Generates HTML data profiling reports
├── generate_visuals.py            # Creates static visualizations for analysis
├── visual_analysis_script.py      # Advanced visual analysis script
├── verify_dashboard.py            # Utility to verify Power BI dashboard data
├── advanced_analysis.py           # Additional advanced analysis scripts
│
├── Master_Cleaned_Sales_Data.csv  # The final output dataset after cleaning
├── Cleaned_Global_Sales.csv       # (Alternative/Intermediate cleaned dataset)
├── Project_Data_Profiling_Report.html # Automated data quality report
├── MID.pbix                       # Power BI Dashboard file
│
├── MODELS/                        # Directory for ML model outputs (performance plots, summaries)
├── VISUALS/                       # Directory for generated charts and graphs
└── ADVANCED_ANALYSIS/             # Directory for advanced analysis outputs
```

## 🛠️ Setup & Installation

1. **Clone the repository** (if applicable) or navigate to the project folder.
2. **Install Dependencies**:
   This project requires Python 3.x and the following libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
   ```
   *Note: `ydata-profiling` may be required for `generate_profiling.py`.*

## 🚀 Usage

### 1. Data Cleaning
Start by running the data cleaning script to generate the master dataset from raw files (`FactSale.csv`, `DimCustomer.csv`, etc.).
```bash
python clean_data.py
```
*Output: `Master_Cleaned_Sales_Data.csv`*

### 2. Machine Learning Models
Run the predictive models to generate insights and performance metrics in the `MODELS/` directory.

- **Sales Prediction (Linear Regression):**
  Predicts `Total Including Tax` based on quantity and tax rates.
  ```bash
  python sales_prediction_model.py
  ```

- **Customer Classification (Logistic Regression):**
  Classifies customers as "High Value" or not.
  ```bash
  python customer_segmentation_model.py
  ```

### 3. Analysis & Visualization
Generate profiling reports and visual analysis charts.
```bash
python generate_profiling.py
python generate_visuals.py
```

## 📈 Key Features

- **Automated Data Pipeline**: Handles missing values, performs type conversion, and merges star-schema tables (Fact & Dimensions).
- **Feature Engineering**: Calculates `Profit Margin`, `Customer Segment`, and extracts date features (Year, Month, Day).
- **Predictive Analytics**:
    - **Regression**: R-squared and MSE metrics for sales forecasting.
    - **Classification**: Confusion matrix and accuracy reports for customer value prediction.
- **Reporting**: Generates comprehensive HTML profiling reports and static plots (`.png`).
- **Dashboarding**: Includes a Power BI (`.pbix`) file for interactive data exploration.

## 📝 Outputs
- **Data**: `Master_Cleaned_Sales_Data.csv`
- **Models**: Performance summaries (`.txt`) and plots (`.png`) in `MODELS/`
- **Reports**: `Project_Data_Profiling_Report.html`
