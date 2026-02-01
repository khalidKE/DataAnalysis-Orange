# Midterm Project: Global Sales Analysis
# This script is optimized for Google Colab/Local Python environment

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import os
import sys

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALS = True
except ImportError:
    HAS_VISUALS = False
    print("⚠️ تحذير: مكتبات الرسم البياني (matplotlib/seaborn) غير متوفرة حالياً.")
    print("سيتم تنفيذ عملية تنظيف البيانات وحساب الإحصائيات فقط.")

# 1. Load Data
def load_data():
    fact_sales = pd.read_csv('FactSale.csv')
    dim_customer = pd.read_csv('DimCustomer.csv', skiprows=1)
    dim_product = pd.read_csv('DimStockItem.csv', skiprows=1)
    dim_city = pd.read_csv('DimCity.csv')
    dim_date = pd.read_csv('DimDate.csv')
    
    # Optional Employee data handling
    if os.path.exists('DimEmployee.xlsx'):
        dim_employee = pd.read_excel('DimEmployee.xlsx')
    else:
        dim_employee = pd.DataFrame()
        
    return fact_sales, dim_customer, dim_product, dim_city, dim_date, dim_employee

# 2. Cleaning & Engineering
def process_data(fact, customer, product, city, date, employee):
    # Duplicates
    fact = fact.drop_duplicates()
    
    # Missing values
    fact['Quantity'] = fact['Quantity'].fillna(0)
    fact['Total Including Tax'] = fact['Total Including Tax'].fillna(fact['Total Including Tax'].median())
    fact['Profit'] = fact['Profit'].fillna(0)
    
    # Date conversion
    fact['Invoice Date Key'] = pd.to_datetime(fact['Invoice Date Key'], errors='coerce')
    
    # Feature Engineering
    fact['Profit Margin'] = np.where(fact['Total Including Tax'] != 0, fact['Profit'] / fact['Total Including Tax'], 0)
    fact['Customer Segment'] = fact['Total Including Tax'].apply(lambda x: 'High' if x > 5000 else ('Medium' if x > 1000 else 'Standard'))
    
    # Merging
    merged = fact.merge(product[['Stock Item Key', 'Stock Item', 'Color']], on='Stock Item Key', how='left')
    merged = merged.merge(city[['City Key', 'City', 'State Province']], on='City Key', how='left')
    merged = merged.merge(customer[['Customer Key', 'Customer']], on='Customer Key', how='left')
    
    return merged

# 3. Main execution
print("Starting analysis...")
fact, customer, product, city, date, employee = load_data()
df = process_data(fact, customer, product, city, date, employee)

# Save cleaned data
df.to_csv('Cleaned_Global_Sales.csv', index=False)
print("Saved cleaned data to Cleaned_Global_Sales.csv")

# 4. Visualizations
if HAS_VISUALS:
    try:
        plt.style.use('ggplot')

        # Plot 1: Top 10 Products
        plt.figure(figsize=(12, 6))
        df.groupby('Stock Item')['Total Including Tax'].sum().sort_values(ascending=False).head(10).plot(kind='bar', color='teal')
        plt.title('Top 10 Products by Sales')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_10_products.png')
        plt.show()

        # Plot 2: Sales by Customer Segment
        plt.figure(figsize=(8, 8))
        df.groupby('Customer Segment')['Total Including Tax'].sum().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Sales Distribution by Customer Segment')
        plt.savefig('sales_by_segment.png')
        plt.show()
        print("✅ تم توليد الصور البيانية (top_10_products.png, sales_by_segment.png)")
    except Exception as e:
        print(f"⚠️ فشل توليد الرسوم البيانية: {e}")
else:
    print("\n💡 نصيحة: للحصول على الرسوم البيانية داخل Python، يمكنك تشغيل هذا الكود في Google Colab.")
    print("أو يمكنك استخدام ملف 'Cleaned_Global_Sales.csv' مباشرة في Power BI لإنشاء لوحات التحكم.")

print("\n--- Summary Statistics ---")
print(f"Total Sales: {df['Total Including Tax'].sum():,.2f}")
print(f"Total Profit: {df['Profit'].sum():,.2f}")
print(f"Total Records: {len(df)}")
