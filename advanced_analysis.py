import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')
sns.set_palette("viridis")

print("⏳ Loading Master Cleaned Data...")
df = pd.read_csv('Master_Cleaned_Sales_Data.csv')
df['Invoice Date Key'] = pd.to_datetime(df['Invoice Date Key'])

if not os.path.exists('ADVANCED_ANALYSIS'):
    os.makedirs('ADVANCED_ANALYSIS')

print("📊 Performing Advanced Data Analysis...")

plt.figure(figsize=(12, 10))
numerical_cols = df.select_dtypes(include=[np.number]).columns
                                      
cols_to_corr = [c for c in numerical_cols if 'Key' not in c and 'ID' not in c]
corr = df[cols_to_corr].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('ADVANCED_ANALYSIS/correlation_heatmap.png')

plt.figure(figsize=(12, 6))
sns.barplot(x='Sales Territory', y='Total Including Tax', data=df, estimator=sum, ci=None)
plt.title('Total Sales by Territory')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ADVANCED_ANALYSIS/sales_by_territory.png')

monthly_sales = df.groupby(df['Invoice Date Key'].dt.to_period('M'))['Total Including Tax'].sum()
growth_rate = monthly_sales.pct_change() * 100
plt.figure(figsize=(12, 6))
growth_rate.plot(kind='line', marker='s', color='orange')
plt.title('Monthly Sales Growth Rate (%)')
plt.axhline(0, color='red', linestyle='--')
plt.ylabel('Growth Percentage')
plt.tight_layout()
plt.savefig('ADVANCED_ANALYSIS/monthly_growth_rate.png')

color_profit = df.groupby('Color')['Profit'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
color_profit.plot(kind='bar', color='purple')
plt.title('Average Profit per Product Color')
plt.ylabel('Avg Profit')
plt.tight_layout()
plt.savefig('ADVANCED_ANALYSIS/color_profitability.png')

top_customers = df.groupby('Customer')['Total Including Tax'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
top_customers.plot(kind='barh', color='teal')
plt.title('Top 10 High-Value Customers')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('ADVANCED_ANALYSIS/top_10_customers.png')

print("✅ Advanced Analysis Complete. Visuals saved in 'ADVANCED_ANALYSIS' folder.")
