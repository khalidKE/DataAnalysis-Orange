import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

df = pd.read_csv('Master_Cleaned_Sales_Data.csv')

if not os.path.exists('MODELS'):
    os.makedirs('MODELS')

print("🤖 Starting Sales Prediction Model (Linear Regression)...")

X = df[['Quantity', 'Unit Price', 'Tax Rate']]
y = df['Total Including Tax']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"✅ Model Trained. R2 Score: {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted Sales (R2: {r2:.4f})')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.tight_layout()
plt.savefig('MODELS/regression_performance.png')

with open('MODELS/linear_regression_summary.txt', 'w') as f:
    f.write("Linear Regression Summary\n")
    f.write("=========================\n")
    f.write(f"R-squared Score: {r2:.4f}\n")
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write("\nCoefficients:\n")
    for col, coef in zip(X.columns, model.coef_):
        f.write(f"- {col}: {coef:.4f}\n")
    f.write(f"Intercept: {model.intercept_:.4f}\n")

print("📊 Results saved in 'MODELS' folder.")
