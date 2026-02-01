import pandas as pd
from ydata_profiling import ProfileReport
import os

# التحقق من وجود الملف
file_path = 'Master_Cleaned_Sales_Data.csv'
if not os.path.exists(file_path):
    print(f"⚠️ الملف {file_path} غير موجود!")
else:
    print("⏳ جاري توليد التقرير الشامل (Auto-EDA)...")
    
    # تحميل البيانات
    df = pd.read_csv(file_path)
    
    # إنشاء تقرير الـ Profiling
    profile = ProfileReport(df, title="Global Sales Data Profiling Report", explorative=True)
    
    # حفظ التقرير كملف HTML
    output_report = "Project_Data_Profiling_Report.html"
    profile.to_file(output_report)
    
    print(f"🎉 تم توليد التقرير بنجاح: {output_report}")
    print("يمكنك الآن فتح هذا الملف في أي متصفح لرؤية تحليل شامل وتلقائي للبيانات.")
