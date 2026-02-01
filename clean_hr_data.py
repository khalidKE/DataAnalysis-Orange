import pandas as pd
import numpy as np
from datetime import datetime

try:
    df = pd.read_csv('HR_Dataset.csv', low_memory=False)

    print("Dataset loaded successfully.")
    print(f"Original shape: {df.shape}")

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    date_columns = ['StartDate', 'ExitDate', 'DOB', 'Survey Date', 'Training Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d-%b-%y', errors='coerce')

   
    categorical_cols = ['Supervisor', 'ADEmail', 'BusinessUnit', 'EmployeeStatus', 'EmployeeType', 'PayZone', 'EmployeeClassificationType', 'TerminationType', 'TerminationDescription', 'DepartmentType', 'Division', 'State', 'JobFunctionDescription', 'GenderCode', 'LocationCode', 'RaceDesc', 'MaritalDesc', 'Performance Score', 'Training Program Name', 'Training Type', 'Training Outcome', 'Location', 'Trainer']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    numeric_cols = ['Current Employee Rating', 'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score', 'Training Duration(Days)', 'Training Cost']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    if 'GenderCode' in df.columns:
        df['GenderCode'] = df['GenderCode'].str.capitalize()

    marital_map = {'Single': 'Single', 'Married': 'Married', 'Divorced': 'Divorced', 'Widowed': 'Widowed', 'Separated': 'Separated'}
    if 'MaritalDesc' in df.columns:
        df['MaritalDesc'] = df['MaritalDesc'].map(marital_map).fillna('Unknown')

    performance_map = {'Fully Meets': 'Fully Meets', 'Exceeds': 'Exceeds', 'Needs Improvement': 'Needs Improvement', 'PIP': 'PIP'}
    if 'Performance Score' in df.columns:
        df['Performance Score'] = df['Performance Score'].map(performance_map).fillna('Unknown')

    df.drop_duplicates(inplace=True)

    if 'StartDate' in df.columns and 'ExitDate' in df.columns:
        df['Tenure'] = (df['ExitDate'] - df['StartDate']).dt.days
        df.loc[df['Tenure'] < 0, 'Tenure'] = np.nan

    if 'DOB' in df.columns:
        df['Age'] = (datetime.now() - df['DOB']).dt.days // 365
        df.loc[df['Age'] < 0, 'Age'] = np.nan

    text_cols = ['FirstName', 'LastName', 'Title', 'Supervisor', 'ADEmail', 'TerminationDescription', 'JobFunctionDescription']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    if 'Employee ID' in df.columns:
        df['Employee ID'] = pd.to_numeric(df['Employee ID'], errors='coerce')
        df.dropna(subset=['Employee ID'], inplace=True)
        df['Employee ID'] = df['Employee ID'].astype(int)

    df.to_csv('HR_Dataset_Cleaned.csv', index=False)

    print("Data cleaning completed. Cleaned data saved to 'HR_Dataset_Cleaned.csv'")
    print(f"Cleaned shape: {df.shape}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting basic cleaning with sed and awk...")
    
    import subprocess
    try:
        subprocess.run(['sed', 's/  */ /g', 'HR_Dataset.csv'], stdout=open('HR_Dataset_Cleaned.csv', 'w'))
        print("Basic cleaning done with sed.")
    except:
        print("Sed not available. Manual cleaning needed.")
