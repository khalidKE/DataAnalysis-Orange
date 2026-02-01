import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8')
# Set font for Arabic if possible, but for charts we'll stick to English labels for compatibility
plt.rcParams['figure.figsize'] = (10, 6)

# Load cleaned data
df = pd.read_csv('Master_Cleaned_Sales_Data.csv')

# Create a folder for images
if not os.path.exists('VISUALS'):
    os.makedirs('VISUALS')

# 1. Sales by Product Category (Top 10)
# Use 'Stock Item' which is now in Title Case
top_products = df.groupby('Stock Item')['Total Including Tax'].sum().sort_values(ascending=False).head(10)
plt.figure()
top_products.plot(kind='barh', color='skyblue')
plt.title('Top 10 Products by Total Sales (Master Data)')
plt.xlabel('Sales Amount')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig('VISUALS/top_10_products.png')

# 2. Sales by State/Province
top_states = df.groupby('State Province')['Total Including Tax'].sum().sort_values(ascending=False).head(10)
plt.figure()
top_states.plot(kind='pie', autopct='%1.1f%%', startangle=140, cmap='viridis')
plt.title('Sales Distribution by State')
plt.ylabel('')
plt.savefig('VISUALS/sales_by_state.png')

# 3. Monthly Sales Trend (Using Bonus Columns)
# Since we already have 'Year' and 'Month' from the cleaning script
# We can sort by date to ensure the trend is correct
df['Invoice Date Key'] = pd.to_datetime(df['Invoice Date Key'])
monthly_sales = df.groupby(df['Invoice Date Key'].dt.to_period('M'))['Total Including Tax'].sum()
plt.figure()
monthly_sales.plot(kind='line', marker='o', color='green')
plt.title('Monthly Sales Trend (Analysis-Ready)')
plt.xlabel('Month')
plt.ylabel('Sales Amount')
plt.grid(True)
plt.savefig('VISUALS/monthly_trend.png')

# 4. Profit Margin by Customer Segment
plt.figure()
sns.boxplot(x='Customer Segment', y='Profit Margin', data=df)
plt.title('Profit Margin Distribution by Customer Segment')
plt.savefig('VISUALS/profit_margin_segments.png')

print("✅ Visuals generated in 'VISUALS' folder")
