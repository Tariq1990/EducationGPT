import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw"))

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
        path = os.path.join(RAW_DIR, f)
        if os.path.exists(path):
            get_columns_robustly(path)
        else:
            print(f"File {path} not found.")
