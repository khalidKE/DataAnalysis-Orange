import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

file_path = 'HR_Dataset.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded.")
except:
    print("Error loading dataset.")
    exit()

df['Turnover'] = df['ExitDate'].notnull().astype(int)
if 'TerminationType' in df.columns:
    df['Turnover'] = df['Turnover'] | (df['TerminationType'].map(lambda x: str(x).lower() not in ['nan', 'unk', 'None', '', 'unkown']))
df['Turnover'] = df['Turnover'].astype(int)
print(f"Turnover Count: {df['Turnover'].sum()} out of {len(df)}")

df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
df['ExitDate'] = pd.to_datetime(df['ExitDate'], errors='coerce')
ref_date = df['ExitDate'].max()
if pd.isnull(ref_date): ref_date = pd.Timestamp.now()
df['TenureDays'] = df.apply(lambda x: (x['ExitDate'] - x['StartDate']).days if pd.notnull(x['ExitDate']) else (ref_date - x['StartDate']).days, axis=1)
df.loc[df['TenureDays'] < 0, 'TenureDays'] = 0

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

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

X.fillna(X.median(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

with open('analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write("--- Model Results ---\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f.write(f"\n{name}:\nAccuracy: {acc:.4f}, F1: {f1:.4f}\n")
        if name == "Random Forest":
            imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            f.write(f"Top Features: {imp.head(5).to_dict()}\n")

    f.write("\n--- Analytics ---\n")
    perf_map = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
    if 'Performance Score' in df.columns:
        df['PerformanceNum'] = df['Performance Score'].map(perf_map)
        avg_perf = df.groupby('Training Outcome')['PerformanceNum'].mean().sort_values(ascending=False)
        f.write(f"Avg Perf by Training Outcome:\n{avg_perf}\n")

    if 'Satisfaction Score' in df.columns:
        top_dept = df.groupby('DepartmentType')['Satisfaction Score'].mean().sort_values(ascending=False).head(3)
        f.write(f"Top Dept Satisfaction:\n{top_dept}\n")
