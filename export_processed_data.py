
import pandas as pd
import numpy as np

file_path = 'HR_Dataset.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


print("Calculating Turnover...")
df['Turnover'] = df['ExitDate'].notnull().astype(int)
if 'TerminationType' in df.columns:
    df['Turnover'] = df['Turnover'] | (df['TerminationType'].map(lambda x: str(x).lower() not in ['nan', 'unk', 'None', '', 'unkown']))
df['Turnover'] = df['Turnover'].astype(int)


print("Calculating Tenure...")
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


for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)


output_path = 'HR_Dataset_Processed.csv'
df.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
