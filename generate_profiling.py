import pandas as pd
from ydata_profiling import ProfileReport
import os

file_path = 'Master_Cleaned_Sales_Data.csv'
if not os.path.exists(file_path):

else:


    df = pd.read_csv(file_path)

    profile = ProfileReport(df, title="Global Sales Data Profiling Report", explorative=True)

    output_report = "Project_Data_Profiling_Report.html"
    profile.to_file(output_report)
    
