import pandas as pd
from openpyxl import load_workbook
import os

files = [
    'Infrastructure.xlsx',
    'Furniture (1).xlsx',
    'Census TeacherInfo.xlsx'
]

def get_columns_robustly(filename):
    print(f"\n--- {filename} ---")
    try:
        # Try pandas with calamine
        df = pd.read_excel(filename, nrows=2, engine='calamine')
        print("Columns:", df.columns.tolist())
        print("First row Sample:", df.iloc[0].to_dict() if len(df) > 0 else "Empty")
    except Exception as e:
        print(f"Calamine failed: {e}")


if __name__ == "__main__":
    for f in files:
        if os.path.exists(f):
            get_columns_robustly(f)
        else:
            print(f"File {f} not found.")
