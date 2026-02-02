import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('Master_Cleaned_Sales_Data.csv')

if not os.path.exists('MODELS'):
    os.makedirs('MODELS')

print("🧠 Starting Customer Classification Model (Logistic Regression)...")

df['Is_High_Value'] = (df['Customer Segment'] == 'High Value').astype(int)

X = df[['Quantity', 'Unit Price', 'Profit', 'Total Dry Items', 'Total Chiller Items']]
y = df['Is_High_Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Classification Complete. Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix: High Value Customer Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('MODELS/classification_confusion_matrix.png')

with open('MODELS/logistic_regression_summary.txt', 'w') as f:
    f.write("Logistic Regression Summary (Customer Classification)\n")
    f.write("==================================================\n")
    f.write(f"Accuracy Score: {acc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)

print("📉 Results saved in 'MODELS' folder.")
