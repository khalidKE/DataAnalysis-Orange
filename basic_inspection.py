import csv
import statistics
from collections import Counter
from datetime import datetime

file_path = 'HR_Dataset.csv'

def calculate_tenure(start_str, exit_str):
    if not start_str: return 0
    try:
        start = datetime.strptime(start_str, "%d-%b-%y")
    except: return 0
    
    if exit_str:
        try:
            exit_date = datetime.strptime(exit_str, "%d-%b-%y")
            return (exit_date - start).days
        except: pass
    
    return (datetime.now() - start).days

data = []
with open(file_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    print("Columns found:", reader.fieldnames)
    for row in reader:
        data.append(row)

print(f"\nTotal Records: {len(data)}")

turnover_counts = Counter()
for row in data:
    status = row.get('EmployeeStatus', 'Unk').strip()
    term_type = row.get('TerminationType', 'Unk').strip()
    exit_date = row.get('ExitDate', '').strip()
    
    is_turnover = False
    if exit_date and term_type not in ['Unk', '']:
        is_turnover = True
    elif term_type not in ['Unk', '', 'None']:
        is_turnover = True
        
    turnover_counts[is_turnover] += 1

print("\nTurnover Distribution:")
print(f"Stayed (Active/Future): {turnover_counts[False]}")
print(f"Left (Turnover): {turnover_counts[True]}")

perf_scores = [row['Performance Score'] for row in data if row['Performance Score']]
print("\nPerformance Score Distribution:")
for k, v in Counter(perf_scores).items():
    print(f"{k}: {v}")

sat_scores = []
for row in data:
    try:
        s = float(row['Satisfaction Score'])
        sat_scores.append(s)
    except: pass

if sat_scores:
    print(f"\nSatisfaction Score Stats: Mean={statistics.mean(sat_scores):.2f}, Median={statistics.median(sat_scores)}")

training_outcomes = [row['Training Outcome'] for row in data if row['Training Outcome']]
print("\nTraining Outcome Distribution:")
for k, v in Counter(training_outcomes).items():
    print(f"{k}: {v}")
