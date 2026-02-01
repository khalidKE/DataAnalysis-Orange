import pandas as pd
import numpy as np
import os

# =========================================
# 1. تحميل البيانات
# =========================================
print("⏳ جاري تحميل البيانات...")

# FactSale.csv
fact_sales = pd.read_csv('FactSale.csv')
print("✅ تم تحميل FactSale.csv")

# DimCustomer.csv - skip first row
dim_customer = pd.read_csv('DimCustomer.csv', skiprows=1)
print("✅ تم تحميل DimCustomer.csv (skip 1)")

# DimEmployee.xlsx
if os.path.exists('DimEmployee.xlsx'):
    dim_employee = pd.read_excel('DimEmployee.xlsx')
    print("✅ تم تحميل DimEmployee.xlsx")
else:
    dim_employee = pd.DataFrame()

# DimStockItem.csv - skip first row
dim_product = pd.read_csv('DimStockItem.csv', skiprows=1)
print("✅ تم تحميل DimStockItem.csv (skip 1)")

# DimCity.csv
dim_city = pd.read_csv('DimCity.csv')
print("✅ تم تحميل DimCity.csv")

# DimDate.csv
dim_date = pd.read_csv('DimDate.csv')
print("✅ تم تحميل DimDate.csv")

# =========================================
# 2. تنظيف البيانات
# =========================================
def clean_df(df, name):
    if df.empty: return df
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"🗑️ تم حذف {before - after} صف مكرر من {name}")
    
    # Strip string columns
    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
    return df

fact_sales = clean_df(fact_sales, "FactSale")
dim_customer = clean_df(dim_customer, "DimCustomer")
dim_employee = clean_df(dim_employee, "DimEmployee")
dim_product = clean_df(dim_product, "DimProduct")
dim_city = clean_df(dim_city, "DimCity")
dim_date = clean_df(dim_date, "DimDate")

# =========================================
# 4. معالجة القيم المفقودة في FactSale
# =========================================
if 'Quantity' in fact_sales.columns:
    fact_sales['Quantity'] = fact_sales['Quantity'].fillna(0)
if 'Total Including Tax' in fact_sales.columns:
    fact_sales['Total Including Tax'] = fact_sales['Total Including Tax'].fillna(fact_sales['Total Including Tax'].median())
if 'Profit' in fact_sales.columns:
    fact_sales['Profit'] = fact_sales['Profit'].fillna(0)

# =========================================
# 5. تصحيح أنواع البيانات
# =========================================
if 'Invoice Date Key' in fact_sales.columns:
    fact_sales['Invoice Date Key'] = pd.to_datetime(fact_sales['Invoice Date Key'], errors='coerce')

# =========================================
# 6. هندسة البيانات
# =========================================
fact_sales['Profit Margin'] = np.where(
    fact_sales['Total Including Tax'] != 0,
    fact_sales['Profit'] / fact_sales['Total Including Tax'],
    0
)

def segment_customer(total):
    if total > 5000: return 'High Value'
    elif total > 1000: return 'Medium Value'
    else: return 'Standard'

fact_sales['Customer Segment'] = fact_sales['Total Including Tax'].apply(segment_customer)

# =========================================
# 7. دمج البيانات
# =========================================
# Note: Check column names in dim_product and dim_city
# Based on diagnosis:
# DimStockItem: 'Stock Item Key', 'Stock Item', 'Color'
# DimCity: 'City Key', 'City', 'State Province'

final_report = fact_sales.merge(
    dim_product[['Stock Item Key', 'Stock Item', 'Color']],
    on='Stock Item Key',
    how='left'
)

final_report = final_report.merge(
    dim_city[['City Key', 'City', 'State Province']],
    on='City Key',
    how='left'
)

# Optional: merge with customer and employee if needed for more insights
final_report = final_report.merge(
    dim_customer[['Customer Key', 'Customer']],
    on='Customer Key',
    how='left'
)

if not dim_employee.empty:
    final_report = final_report.merge(
        dim_employee[['Employee Key', 'Employee']],
        left_on='Salesperson Key',
        right_on='Employee Key',
        how='left'
    )

# =========================================
# 8. حفظ وحساب الإحصائيات البسيطة EDA
# =========================================
output_file = 'Cleaned_Global_Sales_Analysis.csv'
final_report.to_csv(output_file, index=False)

print(f"\n🎉 تم تنظيف البيانات ودمجها بنجاح!")
print(f"📊 إجمالي عدد السجلات: {len(final_report)}")
print(f"💰 إجمالي المبيعات: {final_report['Total Including Tax'].sum():,.2f}")
print(f"📈 إجمالي الربح: {final_report['Profit'].sum():,.2f}")

# Top 5 Cities by Sales
top_cities = final_report.groupby('City')['Total Including Tax'].sum().sort_values(ascending=False).head(5)
print("\n🔥 أعلى 5 مدن من حيث المبيعات:")
print(top_cities)

# Top 5 Products by Sales
top_products = final_report.groupby('Stock Item')['Total Including Tax'].sum().sort_values(ascending=False).head(5)
print("\n📦 أعلى 5 منتجات من حيث المبيعات:")
print(top_products)
