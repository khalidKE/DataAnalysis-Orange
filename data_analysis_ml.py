import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

file_path = 'HR_Dataset.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print(f"Shape: {df.shape}")
print("\nColumns:\n", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

print("\nUnique TerminationType:", df['TerminationType'].unique())
print("Unique EmployeeStatus:", df['EmployeeStatus'].unique())

df['Turnover'] = df['ExitDate'].notnull().astype(int)

print("\nTurnover Distribution:\n", df['Turnover'].value_counts())

df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
df['ExitDate'] = pd.to_datetime(df['ExitDate'], errors='coerce')

ref_date = df['ExitDate'].max()
if pd.isnull(ref_date):
    ref_date = pd.Timestamp.now()

df['TenureDays'] = df.apply(
    lambda x: (x['ExitDate'] - x['StartDate']).days if pd.notnull(x['ExitDate']) 
    else (ref_date - x['StartDate']).days, axis=1
)

df.loc[df['TenureDays'] < 0, 'TenureDays'] = 0

cols_to_drop = [
    'Unnamed: 0', 'FirstName', 'LastName', 'StartDate', 'ExitDate', 'Title', 'Supervisor', 
    'ADEmail', 'EmployeeID', 'Employee ID', 'TerminationDescription', 'DOB', 'Location', 'Trainer', 
    'TerminationType', 'EmployeeStatus', 'Survey Date', 'Training Date' 
]

features = [
    'BusinessUnit', 'PayZone', 'EmployeeType', 'EmployeeClassificationType', 
    'DepartmentType', 'Division', 'State', 'JobFunctionDescription', 
    'GenderCode', 'LocationCode', 'RaceDesc', 'MaritalDesc', 
    'Performance Score', 'Current Employee Rating', 
    'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score',
    'Training Program Name', 'Training Type', 'Training Outcome', 
    'Training Duration(Days)', 'Training Cost', 'TenureDays'
]

features = [c for c in features if c in df.columns]

X = df[features].copy()
y = df['Turnover']

categorical_cols = X.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_cols.tolist())

le = LabelEncoder()
for col in categorical_cols:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}

print("\n--- Model Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\n--- Summary of Accuracies ---")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
