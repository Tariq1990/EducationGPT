import pandas as pd
import numpy as np
import os

def load_and_merge_data(data_dir):
    """
    Loads Infrastructure, Furniture, and TeacherInfo datasets and merges them.
    """
    infra_path = os.path.join(data_dir, 'Infrastructure.xlsx')
    furn_path = os.path.join(data_dir, 'Furniture (1).xlsx')
    teacher_path = os.path.join(data_dir, 'Census TeacherInfo.xlsx')
    non_teacher_path = os.path.join(data_dir, 'Census NonteacherInfo.xlsx')
    ptc_path = os.path.join(data_dir, 'Census Ptc Info.xlsx')

    print("Loading data...")
    df_infra = pd.read_excel(infra_path, engine='calamine')
    df_furn = pd.read_excel(furn_path, engine='calamine')
    df_teach = pd.read_excel(teacher_path, engine='calamine')
    df_non_teach = pd.read_excel(non_teacher_path, engine='calamine')
    df_ptc = pd.read_excel(ptc_path, engine='calamine')

    # 1. Process Teacher Info (Aggregate by EMIS Code)
    print("Processing Teacher Info...")
    teacher_agg = df_teach.groupby('emiscode').agg(
        total_teachers=('personalNo', 'count'),
        avg_bps_teachers=('BPS', 'mean'),
        qualified_teachers=('H_QualificationLevel', lambda x: x.isin(['Masters', 'M.Phil', 'Ph.D', 'Post Graduate']).sum())
    ).reset_index()
    teacher_agg.rename(columns={'emiscode': 'EMIS Code'}, inplace=True)

    # 2. Process Non-Teacher Info
    print("Processing Non-Teacher Info...")
    non_teacher_agg = df_non_teach.groupby('EMISCode').size().reset_index(name='total_non_teachers')
    non_teacher_agg.rename(columns={'EMISCode': 'EMIS Code'}, inplace=True)

    # 3. Process PTC Info
    print("Processing PTC Info...")
    # We take the most recent record if duplicates exist, or just merge if unique
    ptc_subset = df_ptc[['EMISCode', 'Presentbalance', 'Fund Recieved from Govt in This Year']].copy()
    ptc_subset.rename(columns={'EMISCode': 'EMIS Code'}, inplace=True)
    ptc_agg = ptc_subset.groupby('EMIS Code').agg({
        'Presentbalance': 'sum',
        'Fund Recieved from Govt in This Year': 'sum'
    }).reset_index()

    # 4. Merge Data
    print("Merging datasets...")
    df_merged = pd.merge(df_infra, df_furn, on='EMIS Code', how='left', suffixes=('', '_furn'))
    df_final = pd.merge(df_merged, teacher_agg, on='EMIS Code', how='left')
    df_final = pd.merge(df_final, non_teacher_agg, on='EMIS Code', how='left')
    df_final = pd.merge(df_final, ptc_agg, on='EMIS Code', how='left')

    # 5. Create Target Label: 'is_critical'
    print("Creating target label...")
    
    # Helper to clean comma-separated numbers
    def clean_num(s):
        if pd.isna(s): return 0
        if isinstance(s, str):
            s = s.replace(',', '').strip()
        try:
            return float(s)
        except:
            return 0

    df_final['Presentbalance'] = df_final['Presentbalance'].apply(clean_num)
    df_final['Fund Recieved from Govt in This Year'] = df_final['Fund Recieved from Govt in This Year'].apply(clean_num)
    
    df_final['is_critical'] = (
        (df_final['Whole Building Needs Reconstruction'] == 'Yes') |
        (df_final['No of Class Room Need Reconstruction'] > 0) |
        (df_final['Desks Two Seater New Required'] > 0) |
        (df_final['Fans New Required'] > 0) |
        (df_final['total_teachers'] <= 1) |
        (df_final['Presentbalance'] < 1000) # Budget shortage
    ).astype(int)



    # 6. Clean up columns (Remove duplicates from merge)
    cols_to_drop = [c for c in df_final.columns if '_furn' in c]
    df_final.drop(columns=cols_to_drop, inplace=True)

    return df_final

def preprocess_features(df):
    """
    Basic cleaning and feature selection.
    """
    # Select expanded features
    features = [
        'Gender', 'LEVEL', 'District', 'BuildingOwnerShip', 
        'TotalLand_Allocated', 'Covered_Area', 'UnCovered_Area',
        'No of Pakka Class Rooms', 'No. Of Pakka Other Rooms',
        'Desks Two Seater Useable', 'Tablet Chairs Useable', 'Fans Useable',
        'total_teachers', 'avg_bps_teachers', 'qualified_teachers',
        'total_non_teachers', 'Presentbalance', 'Fund Recieved from Govt in This Year'
    ]

    
    # Ensure target is present
    if 'is_critical' in df.columns:
        y = df['is_critical']
        X = df[features]
        return X, y
    else:
        return df[features]

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.getenv(
        "DATA_DIR",
        os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw"))
    )
    df = load_and_merge_data(DATA_DIR)
    print(f"Final Data Shape: {df.shape}")
    print(f"Critical Schools: {df['is_critical'].sum()} out of {len(df)}")
    
    X, y = preprocess_features(df)
    print("Features selected.")
    
    # Save processed data for training
    out_path = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "processed_school_data.csv"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
