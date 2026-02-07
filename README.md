

# Machine Learning with Scikit-Learn (Regression & Classification)

This project demonstrates two core machine learning workflows using **Python** and **scikit-learn**:

1. **Simple Linear Regression** on synthetic data
2. **Binary Classification (Breast Cancer Detection)** using Logistic Regression

The notebook covers data preprocessing, visualization, model training, evaluation, and interpretation.

---

## Technologies Used

* Python 3.12
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

> ⚠️ On Windows, you may need to run pip with `--user` or as administrator if you face permission errors.

---

## Part 1: Linear Regression (Synthetic Data)

### Description

A simple linear regression model is trained on artificially generated data to predict a linear relationship with noise.

### Steps

* Generate input data using NumPy
* Add random noise to simulate real-world data
* Scale features using `MinMaxScaler`
* Train a `LinearRegression` model inside a pipeline
* Evaluate model performance using R² score
* Make predictions on new data

### Example Output

* **Model Score (R²):** ~0.86
* **Prediction for x = 9:** ~26.9

---

## Part 2: Breast Cancer Classification

### Dataset

* **File:** `Cancer_data.csv`
* **Target:** `diagnosis`

  * `M` → Malignant
  * `B` → Benign

### Data Overview

* 569 samples
* 30 numerical features
* Binary classification problem

---

## Exploratory Data Analysis (EDA)

* Class distribution visualization
* Scatter plots comparing malignant vs benign tumors
* Pair plots for selected features:

  * radius_mean
  * texture_mean
  * perimeter_mean
  * area_mean

---

## Data Preprocessing

* Encode diagnosis labels:

  * Benign → `0`
  * Malignant → `1`
* Drop non-informative column (`diagnosis`)
* Train/Test split (80% / 20%)
* Feature scaling using `MinMaxScaler`

---

## Logistic Regression Model

### Training

* Model: `LogisticRegression`
* Max iterations: `1000`

### Evaluation Metrics

* **Accuracy**
* **Confusion Matrix**
* Heatmap visualization for predictions vs actual values

### Results

* **Test Accuracy:** ~96.5%
* **Confusion Matrix:**

```
[[73  1]
 [ 3 37]]
```

This indicates strong performance with very few misclassifications.

---

## Visualizations

* Scatter plots for feature comparison
* Pair plots for multivariate relationships
* Confusion matrix heatmap

---

## Project Structure

```
├── README.md
├── Cancer_data.csv
└── notebook.ipynb
```

---

## Key Learnings

* Feature scaling is critical for regression and classification
* Logistic Regression performs well on well-structured medical datasets
* Visualization helps clearly separate malignant and benign cases
* Pipelines simplify preprocessing + modeling

---


Machine Learning & Data Analysis Project

