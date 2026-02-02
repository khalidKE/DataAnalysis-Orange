import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
                                                                                                 
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('Master_Cleaned_Sales_Data.csv')

if not os.path.exists('VISUALS'):
    os.makedirs('VISUALS')

top_products = df.groupby('Stock Item')['Total Including Tax'].sum().sort_values(ascending=False).head(10)
plt.figure()
top_products.plot(kind='barh', color='skyblue')
plt.title('Top 10 Products by Total Sales (Master Data)')
plt.xlabel('Sales Amount')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig('VISUALS/top_10_products.png')

top_states = df.groupby('State Province')['Total Including Tax'].sum().sort_values(ascending=False).head(10)
plt.figure()
top_states.plot(kind='pie', autopct='%1.1f%%', startangle=140, cmap='viridis')
plt.title('Sales Distribution by State')
plt.ylabel('')
plt.savefig('VISUALS/sales_by_state.png')

df['Invoice Date Key'] = pd.to_datetime(df['Invoice Date Key'])
monthly_sales = df.groupby(df['Invoice Date Key'].dt.to_period('M'))['Total Including Tax'].sum()
plt.figure()
monthly_sales.plot(kind='line', marker='o', color='green')
plt.title('Monthly Sales Trend (Analysis-Ready)')
plt.xlabel('Month')
plt.ylabel('Sales Amount')
plt.grid(True)
plt.savefig('VISUALS/monthly_trend.png')

plt.figure()
sns.boxplot(x='Customer Segment', y='Profit Margin', data=df)
plt.title('Profit Margin Distribution by Customer Segment')
plt.savefig('VISUALS/profit_margin_segments.png')

print("✅ Visuals generated in 'VISUALS' folder")
