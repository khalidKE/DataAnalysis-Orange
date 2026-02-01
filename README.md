# HR Data Analysis & Machine Learning

This project performs data processing, exploratory analysis, and machine learning to predict employee turnover and analyze performance drivers using the `HR_Dataset.csv`.

## Files
- `refined_analysis.py`: Main Python script for data processing, ML modeling (Logistic Regression, Random Forest, Gradient Boosting), and specific HR analytics.
- `basic_inspection.py`: A dependency-free script to inspect the raw CSV data.
- `HR_Dataset.csv`: The source dataset.

## Setup & Running
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Run the analysis:
   ```bash
   python refined_analysis.py
   ```

## Key features
- **Turnover Prediction**: Uses classification models to predict if an employee will leave.
- **Performance Analysis**: Analyzes the impact of training on performance scores.
- **Department Insights**: Identifies departments with highest satisfaction and engagement.

## Methodology
- **Data Cleaning**: Dates are converted to datetime objects, tenure is calculated, and categorical variables are encoded.
- **Feature Engineering**: `TenureDays` is derived from Start and Exit dates.
- **Handling Imbalance**: Models use `class_weight='balanced'` to handle the uneven distribution of turnover.
- **Evaluation**: specific focus on F1-Score and Feature Importance.

## Insights (Preliminary)
Based on data inspection:
- Turnover is a significant event but less frequent than retention (Imbalanced dataset).
- Training outcomes vary significantly, with a mix of Failed, Incomplete, Passed, and Completed.
- Performance scores are categorical (Fully Meets, Exceeds, Needs Improvement, PIP).

## Questions Answered
- **Factors predicting turnover**: The Random Forest model outputs feature importance to identify these.
- **Training effectiveness**: The script correlates training outcomes with performance scores.
- **Department Sat/Eng**: aggregated statistics per department.
