import pandas as pd

df = pd.read_csv('Master_Cleaned_Sales_Data.csv')

total_sales = df['Total Including Tax'].sum()
total_profit = df['Profit'].sum()
sales_count = len(df)
avg_profit_margin = df['Profit Margin'].mean()

print("--- Verification Results ---")
print(f"1. Total Sales (Expected: 22.86M): {total_sales:,.2f} ({total_sales/1e6:.2f}M)")
print(f"2. Total Profit (Expected: 9.92M): {total_profit:,.2f} ({total_profit/1e6:.2f}M)")
print(f"3. Sales Count (Expected: 26.397K): {sales_count:,} ({sales_count/1e3:.3f}K)")
print(f"4. Avg Profit Margin (Expected: 0.47): {avg_profit_margin:.4f}")

top_customers = df.groupby('Customer')['Total Including Tax'].sum().sort_values(ascending=False).head(5)
print("\n--- Top Customers (Verification) ---")
print(top_customers)
