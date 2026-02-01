
import pandas as pd

try:
    df = pd.read_csv('HR_Dataset.csv')
    print("Columns:", df.columns.tolist())
    print("\nTotal Rows:", len(df))
    

    cat_cols = df.select_dtypes(include=['object']).columns
    print("\nCategorical Cardinality:")
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        
    print("\nDate Sample:")
    print("DOB:", df['DOB'].head(3).tolist())
    print("StartDate:", df['StartDate'].head(3).tolist())
    
except Exception as e:
    print(e)
