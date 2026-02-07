# Loan Data Analysis & Profiling 

## Overview

This project performs **exploratory data analysis (EDA)** on a loan dataset (`loans.csv`) using Python.
It covers data cleaning, type conversion, descriptive statistics, visualization, correlation analysis, and automated data profiling using **ydata-profiling**.

The goal is to better understand loan characteristics, repayment behavior, interest rates, and relationships between numeric variables.

---

## Dataset Description

The dataset contains **443 loan records** with the following columns:

| Column Name   | Description                              |
| ------------- | ---------------------------------------- |
| `client_id`   | Unique identifier for clients            |
| `loan_type`   | Type of loan (home, credit, cash, other) |
| `loan_amount` | Loan amount issued                       |
| `repaid`      | Loan repayment status (0 = No, 1 = Yes)  |
| `loan_id`     | Unique loan identifier                   |
| `loan_start`  | Loan start date                          |
| `loan_end`    | Loan end date                            |
| `rate`        | Interest rate                            |

* Shape: **(443, 8)**
* No missing values detected

---

## Technologies Used 🛠

* **Python 3.12**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **ydata-profiling**
* **ipywidgets** (for notebook rendering)

---

## Key Steps Performed

### 1. Data Loading & Inspection

* Loaded dataset using `pandas.read_csv`
* Checked structure with `.head()`, `.shape()`, `.info()`
* Verified absence of missing values

### 2. Data Cleaning & Type Conversion

* Converted:

  * `loan_id` → `object`
  * `repaid` → `category`
  * `loan_start`, `loan_end` → `datetime`
* Ensured correct data types for analysis

### 3. Descriptive Statistics

* Generated numerical summaries using `df.describe()`
* Analyzed categorical variables with `describe(include='object')`

### 4. High-Value Loan Analysis

* Filtered loans with amounts **above the dataset mean**
* Explored patterns among high-value loans

### 5. Group Analysis

* Computed **average loan amount by loan type**

```text
cash      8098
credit    7429
home      8009
other     8388
```

### 6. Data Visualization 📈

* Box plots for:

  * Loan amount
  * Interest rate
* Histogram showing loan amount distribution
* Correlation heatmap between numeric variables

### 7. Correlation Analysis

* Calculated correlation matrix for numeric features
* Visualized using Seaborn heatmap
* Found weak correlations between loan amount, rate, and client ID

---

## Automated Data Profiling 🤖

Used **ydata-profiling** to generate a comprehensive HTML report:

* Variable distributions
* Correlations
* Outliers
* Data quality warnings

### Output:

```text
loan_data_report.html
```

> ⚠️ Note: If the report fails to render inside Jupyter, install `ipywidgets` and restart the kernel.

```bash
pip install ipywidgets
```

---

## How to Run 🚀

1. Clone the repository
2. Place `loans.csv` in the project directory
3. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn ydata-profiling ipywidgets
```

4. Run the notebook or script
5. Open `loan_data_report.html` in your browser

---

## Project Outcome

This analysis provides:

* Clear understanding of loan distributions
* Insights into repayment and loan types
* Visual and statistical summaries
* A reusable EDA workflow for financial datasets

