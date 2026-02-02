import pandas as pd
import numpy as np
import os
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALS = True
except ImportError:
    HAS_VISUALS = False

def load_data():
    fact_sales = pd.read_csv('FactSale.csv')
    dim_customer = pd.read_csv('DimCustomer.csv', skiprows=1)
    dim_product = pd.read_csv('DimStockItem.csv', skiprows=1)
    dim_city = pd.read_csv('DimCity.csv')
    
    if os.path.exists('DimEmployee.xlsx'):
        dim_employee = pd.read_excel('DimEmployee.xlsx')
    else:
        dim_employee = pd.DataFrame()
        
    return fact_sales, dim_customer, dim_product, dim_city, dim_employee

def process_data(fact, customer, product, city, employee):
                
    fact = fact.drop_duplicates()

    fact['Quantity'] = fact['Quantity'].fillna(0)
    fact['Total Including Tax'] = fact['Total Including Tax'].fillna(fact['Total Including Tax'].median())
    fact['Profit'] = fact['Profit'].fillna(0)

    for df in [customer, product, city]:
        text_cols = df.select_dtypes(include='object').columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip().str.title()

    fact['Invoice Date Key'] = pd.to_datetime(fact['Invoice Date Key'], errors='coerce')
    fact['Year'] = fact['Invoice Date Key'].dt.year
    fact['Month'] = fact['Invoice Date Key'].dt.month_name()

    fact['Profit Margin'] = np.where(fact['Total Including Tax'] != 0, fact['Profit'] / fact['Total Including Tax'], 0)
    fact['Customer Segment'] = fact['Total Including Tax'].apply(lambda x: 'High Value' if x > 5000 else ('Medium Value' if x > 1000 else 'Standard'))

    fact['Stock Item Key'] = fact['Stock Item Key'].astype(int)
    product['Stock Item Key'] = product['Stock Item Key'].astype(int)

    merged = fact.merge(product[['Stock Item Key', 'Stock Item', 'Color']], on='Stock Item Key', how='left')
    merged = merged.merge(city[['City Key', 'City', 'State Province']], on='City Key', how='left')
    merged = merged.merge(customer[['Customer Key', 'Customer']], on='Customer Key', how='left')
    
    return merged

print("Starting analysis...")
fact, customer, product, city, employee = load_data()
df = process_data(fact, customer, product, city, employee)

df.to_csv('Master_Cleaned_Sales_Data.csv', index=False)
print("Saved cleaned data to Master_Cleaned_Sales_Data.csv")

if HAS_VISUALS:
    try:
        plt.style.use('ggplot')

        plt.figure(figsize=(12, 6))
        df.groupby('Stock Item')['Total Including Tax'].sum().sort_values(ascending=False).head(10).plot(kind='bar', color='teal')
        plt.title('Top 10 Products by Sales')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        df.groupby(df['Invoice Date Key'].dt.to_period('M'))['Total Including Tax'].sum().plot(kind='line', marker='o')
        plt.title('Monthly Sales Trend')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
else:
    print("\n💡 Tip: Run this code in Google Colab to see interactive charts!")

print("\n--- Summary Statistics ---")
print(f"Total Sales: {df['Total Including Tax'].sum():,.2f}")
print(f"Total Profit: {df['Profit'].sum():,.2f}")
print(f"Total Records: {len(df)}")
