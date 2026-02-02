import pandas as pd
import numpy as np
import os



fact_sales = pd.read_csv('FactSale.csv')
dim_customer = pd.read_csv('DimCustomer.csv', skiprows=1)
dim_product = pd.read_csv('DimStockItem.csv', skiprows=1)
dim_city = pd.read_csv('DimCity.csv')
dim_date = pd.read_csv('DimDate.csv')

try:
    if os.path.exists('DimEmployee.xlsx'):
        dim_employee = pd.read_excel('DimEmployee.xlsx')
    elif os.path.exists('DimEmployee.csv'):
        dim_employee = pd.read_csv('DimEmployee.csv')
    else:
                                                                         
        found = False
        for f in os.listdir('.'):
            if 'DimEmployee' in f:
                if f.endswith('.xlsx'):
                    dim_employee = pd.read_excel(f)
                else:
                    dim_employee = pd.read_csv(f)
                found = True
                break
        if not found:
            dim_employee = pd.DataFrame()
         
    if not dim_employee.empty:
        pass
            
except Exception as e:
    dim_employee = pd.DataFrame()


def clean_df(df, name):
    if df.empty: return df

    df = df.drop_duplicates()

    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
                                                                                   
        df[col] = df[col].astype(str).str.strip().str.title()
    
    return df

fact_sales = clean_df(fact_sales, "FactSale")
dim_customer = clean_df(dim_customer, "DimCustomer")
dim_product = clean_df(dim_product, "DimProduct")
dim_city = clean_df(dim_city, "DimCity")

fact_sales['Invoice Date Key'] = pd.to_datetime(fact_sales['Invoice Date Key'], errors='coerce')
fact_sales['Year'] = fact_sales['Invoice Date Key'].dt.year
fact_sales['Month'] = fact_sales['Invoice Date Key'].dt.month_name()
fact_sales['Day_of_Week'] = fact_sales['Invoice Date Key'].dt.day_name()

fact_sales['Profit Margin'] = np.where(
    fact_sales['Total Including Tax'] != 0,
    fact_sales['Profit'] / fact_sales['Total Including Tax'],
    0
)

def get_segment(total):
    if total > 5000: return 'High Value'
    elif total > 1000: return 'Medium Value'
    else: return 'Standard'

fact_sales['Customer Segment'] = fact_sales['Total Including Tax'].apply(get_segment)


keys_to_int = ['Stock Item Key', 'City Key', 'Customer Key']
for key in keys_to_int:
    if key in fact_sales.columns:
        fact_sales[key] = pd.to_numeric(fact_sales[key], errors='coerce').fillna(0).astype(int)
    if key in dim_product.columns and key == 'Stock Item Key':
        dim_product[key] = pd.to_numeric(dim_product[key], errors='coerce').fillna(0).astype(int)
    if key in dim_city.columns and key == 'City Key':
        dim_city[key] = pd.to_numeric(dim_city[key], errors='coerce').fillna(0).astype(int)
    if key in dim_customer.columns and key == 'Customer Key':
        dim_customer[key] = pd.to_numeric(dim_customer[key], errors='coerce').fillna(0).astype(int)

final_report = fact_sales.merge(
    dim_product[['Stock Item Key', 'Stock Item', 'Color']], 
    on='Stock Item Key', how='left'
).merge(
    dim_city[['City Key', 'City', 'State Province', 'Sales Territory']], 
    on='City Key', how='left'
).merge(
    dim_customer[['Customer Key', 'Customer']], 
    on='Customer Key', how='left'
)

if not dim_employee.empty and 'Salesperson Key' in fact_sales.columns:
    dim_employee['Employee Key'] = pd.to_numeric(dim_employee['Employee Key'], errors='coerce').fillna(0).astype(int)
    final_report = final_report.merge(
        dim_employee[['Employee Key', 'Employee']], 
        left_on='Salesperson Key', right_on='Employee Key', how='left'
    )

output_file = 'Master_Cleaned_Sales_Data.csv'
final_report.to_csv(output_file, index=False)

