import pandas as pd
import os
import sys

# Set stdout to utf-8 just in case
sys.stdout.reconfigure(encoding='utf-8')

log_file = 'clean_log.txt'

def log(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)

if os.path.exists(log_file):
    os.remove(log_file)

files = ['FactSale.csv', 'DimCustomer.csv', 'DimStockItem.csv', 'DimCity.csv', 'DimDate.csv']

for f in files:
    try:
        # Read the first line to see if it's junk
        with open(f, 'r', encoding='utf-8', errors='ignore') as temp_f:
            first_line = temp_f.readline()
            log(f"--- File: {f} ---")
            log(f"First line: {first_line.strip()}")
        
        # Determine skiprows
        skip = 0
        if f == 'DimStockItem.csv' and (first_line.startswith(',,') or not first_line.strip()):
            skip = 1
            log("Skipping first row for DimStockItem.csv")

        df = pd.read_csv(f, nrows=5, skiprows=skip)
        log(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        log(f"Error reading {f}: {e}")

if os.path.exists('DimEmployee.xlsx'):
    try:
        df = pd.read_excel('DimEmployee.xlsx', nrows=5)
        log("--- File: DimEmployee.xlsx ---")
        log(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        log(f"Error reading DimEmployee.xlsx: {e}")
