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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

file_path = 'HR_Dataset.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print(f"Shape: {df.shape}")
print("\nTurnover Check:")
df['Turnover'] = df['ExitDate'].notnull().astype(int)

if 'TerminationType' in df.columns:
    df['Turnover'] = df['Turnover'] | (df['TerminationType'].map(lambda x: str(x).lower() not in ['nan', 'unk', 'None', '']))

df['Turnover'] = df['Turnover'].astype(int)
print(df['Turnover'].value_counts())

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

results = {}
feature_importances = {}

print("\n--- Model Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'Accuracy': acc, 'F1 Score': f1}
    
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    if name == "Random Forest":
        feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop 10 Feature Importances (Random Forest):")
        print(feature_importances.head(10))

if not feature_importances.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.head(10), y=feature_importances.head(10).index)
    plt.title('Top 10 Features for Predicting Turnover')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    try:
        plt.savefig('feature_importance.png')
        print("\nFeature importance plot saved as 'feature_importance.png'")
    except:
        pass


print("\n--- HR Specific Analytics ---")

perf_map = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
df['PerformanceNum'] = df['Performance Score'].map(perf_map)
training_impact = df.groupby('Training Outcome')['PerformanceNum'].mean().sort_values(ascending=False)
print("\nAverage Performance Score by Training Outcome:")
print(training_impact)

dept_stats = df.groupby('DepartmentType')[['Satisfaction Score', 'Engagement Score']].mean().sort_values(by='Satisfaction Score', ascending=False)
print("\nDepartment Satisfaction & Engagement (Top 5):")
print(dept_stats.head())

role_turnover = df.groupby('Title')['Turnover'].mean().sort_values(ascending=False)
print("\nTurnover Rate by Job Title (Top 5 Highest):")
print(role_turnover.head())

corr = df[['Training Duration(Days)', 'Training Cost', 'PerformanceNum', 'Turnover']].corr()
print("\nCorrelation Matrix (Training vs Performance/Turnover):")
print(corr)
