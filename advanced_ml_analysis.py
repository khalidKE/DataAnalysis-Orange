
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

file_path = 'HR_Dataset.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded.")
except Exception as e:
    print(f"Error: {e}")
    exit()

df['Turnover'] = df['ExitDate'].notnull().astype(int)
if 'TerminationType' in df.columns:
    df['Turnover'] = df['Turnover'] | (df['TerminationType'].fillna('Unk').apply(lambda x: str(x).lower() not in ['nan', 'unk', 'none', '']))
df['Turnover'] = df['Turnover'].astype(int)


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


df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['Age'] = (ref_date - df['DOB']).dt.days / 365.25
df.loc[df['Age'] < 18, 'Age'] = df['Age'].median() # Fix weird ages

perf_map = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
df['PerformanceNum'] = df['Performance Score'].map(perf_map)
df['PerformanceNum'].fillna(2, inplace=True) # Assume avg if missing

df['StartYear'] = df['StartDate'].dt.year
df['StartMonth'] = df['StartDate'].dt.month

numerical_features = ['TenureDays', 'Age', 'PerformanceNum', 'Engagement Score', 
                      'Satisfaction Score', 'Work-Life Balance Score', 'Training Duration(Days)', 
                      'Training Cost', 'StartYear']

categorical_features = ['BusinessUnit', 'PayZone', 'DepartmentType', 'Division', 'State', 
                        'JobFunctionDescription', 'GenderCode', 'RaceDesc', 'MaritalDesc', 
                        'Training Program Name', 'Training Outcome']


numerical_features = [c for c in numerical_features if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

X = df[numerical_features + categorical_features].copy()
y = df['Turnover']


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

models = {
    
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=200),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced')
}

output_file = 'advanced_results.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("--- Advanced Analysis Results ---\n")
    
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        result_str = f"\nModel: {name}\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n"
        print(result_str)
        f.write(result_str)
        f.write(classification_report(y_test, y_pred))
        f.write("\n" + "="*30 + "\n")

print(f"Results saved to {output_file}")
