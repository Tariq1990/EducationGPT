import os
import numpy as np
import pandas as pd

EMIS_CANDIDATES = ["EMIS Code", "EMISCode", "EmisCode", "emiscode", "EMISCODE"]


def clean_num(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if not value:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def to_flag(value):
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    if s in {"yes", "y", "true", "1", "available", "functional", "working"}:
        return 1
    return 0


def normalize_emis(df):
    if df is None or df.empty:
        return df
    for c in EMIS_CANDIDATES:
        if c in df.columns:
            col = c
            break
    else:
        return df

    out = df.copy()
    out.rename(columns={col: "EMIS Code"}, inplace=True)
    out["EMIS Code"] = (
        out["EMIS Code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    out = out[out["EMIS Code"].notna() & (out["EMIS Code"] != "")]
    return out


def read_xlsx(data_dir, filename):
    path = os.path.join(data_dir, filename)
    df = pd.read_excel(path, engine="calamine")
    return normalize_emis(df)


def aggregate_teacher(df_teach):
    grp = df_teach.groupby("EMIS Code", dropna=False)
    out = grp.agg(
        total_teachers=("personalNo", "count"),
        avg_bps_teachers=("BPS", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        qualified_teachers=(
            "H_QualificationLevel",
            lambda x: x.astype(str).isin(["Masters", "M.Phil", "Ph.D", "Post Graduate"]).sum(),
        ),
    ).reset_index()
    return out


def aggregate_non_teacher(df_non_teach):
    out = (
        df_non_teach.groupby("EMIS Code", dropna=False)
        .size()
        .reset_index(name="total_non_teachers")
    )
    return out


def aggregate_ptc(df_ptc):
    for col in ["Presentbalance", "Fund Recieved from Govt in This Year", "Fund Received from other"]:
        if col in df_ptc.columns:
            df_ptc[col] = df_ptc[col].apply(clean_num)

    agg_map = {}
    for col in ["Presentbalance", "Fund Recieved from Govt in This Year", "Fund Received from other"]:
        if col in df_ptc.columns:
            agg_map[col] = "sum"

    out = df_ptc.groupby("EMIS Code", dropna=False).agg(agg_map).reset_index()
    return out


def aggregate_it_lab(df_it):
    out = pd.DataFrame({"EMIS Code": df_it["EMIS Code"].dropna().unique()})
    grouped = df_it.groupby("EMIS Code", dropna=False)

    out = grouped.agg(
        itlab_available=("ITLab", lambda s: int(any(to_flag(v) for v in s))),
        itlab_functional=("ITLabFunctional", lambda s: int(any(to_flag(v) for v in s))),
        itlab_total_computers=("Total No. of Computers", lambda s: pd.to_numeric(s, errors="coerce").max()),
        itlab_functional_computers=("NoOfFunctionalComputers", lambda s: pd.to_numeric(s, errors="coerce").max()),
        itlab_internet_available=("Internet Available", lambda s: int(any(to_flag(v) for v in s))),
    ).reset_index()
    return out


def aggregate_enrollment(df_enroll):
    class_cols = [c for c in df_enroll.columns if str(c).startswith("Class-") or str(c) in {"Nursery", "Prep"}]
    out = df_enroll.copy()
    for c in class_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    primary_cols = [c for c in ["Nursery", "Prep", "Class-1", "Class-2", "Class-3", "Class-4", "Class-5"] if c in out.columns]
    middle_secondary_cols = [c for c in ["Class-6", "Class-7", "Class-8", "Class-9", "Class-10"] if c in out.columns]
    higher_secondary_cols = [c for c in ["Class-11", "Class-12"] if c in out.columns]

    out["total_enrollment"] = out[class_cols].sum(axis=1) if class_cols else 0
    out["enrollment_primary"] = out[primary_cols].sum(axis=1) if primary_cols else 0
    out["enrollment_middle_secondary"] = out[middle_secondary_cols].sum(axis=1) if middle_secondary_cols else 0
    out["enrollment_higher_secondary"] = out[higher_secondary_cols].sum(axis=1) if higher_secondary_cols else 0

    keep = [
        "EMIS Code",
        "total_enrollment",
        "enrollment_primary",
        "enrollment_middle_secondary",
        "enrollment_higher_secondary",
    ]
    return out.groupby("EMIS Code", dropna=False)[keep[1:]].sum().reset_index()


def aggregate_disabled(df_dis):
    disorder_cols = [c for c in df_dis.columns if "Disorder" in str(c)]
    out = df_dis.copy()
    for c in disorder_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["disabled_students_total"] = out[disorder_cols].sum(axis=1) if disorder_cols else 0
    return out.groupby("EMIS Code", dropna=False)[["disabled_students_total"]].sum().reset_index()


def aggregate_sanctioned_non_teaching(df_nt):
    out = df_nt.copy()
    out["Sanctioned Posts"] = pd.to_numeric(out["Sanctioned Posts"], errors="coerce").fillna(0)
    out["Filled Posts"] = pd.to_numeric(out["Filled Posts"], errors="coerce").fillna(0)

    agg = out.groupby("EMIS Code", dropna=False).agg(
        nonteach_sanctioned_posts=("Sanctioned Posts", "sum"),
        nonteach_filled_posts=("Filled Posts", "sum"),
    ).reset_index()
    agg["nonteach_vacancy_posts"] = (agg["nonteach_sanctioned_posts"] - agg["nonteach_filled_posts"]).clip(lower=0)
    return agg


def aggregate_sanctioned_teaching(df_t):
    out = df_t.copy()
    out["Sanctioned Posts"] = pd.to_numeric(out["Sanctioned Posts"], errors="coerce").fillna(0)
    out["Filled Posts"] = pd.to_numeric(out["Filled Posts"], errors="coerce").fillna(0)

    agg = out.groupby("EMIS Code", dropna=False).agg(
        teach_sanctioned_posts=("Sanctioned Posts", "sum"),
        teach_filled_posts=("Filled Posts", "sum"),
    ).reset_index()
    agg["teach_vacancy_posts"] = (agg["teach_sanctioned_posts"] - agg["teach_filled_posts"]).clip(lower=0)
    return agg


def aggregate_basic_facilities(df_basic):
    col_map = {
        "basic_electric_available": "Electric Availability",
        "basic_electric_functional": "Electric Fuctionality",
        "basic_water_available": "Water Availability",
        "basic_water_functional": "Water Functionality",
        "basic_toilet_available": "Toilet Availability",
        "basic_toilet_functional": "Toilet Functionality",
        "basic_wall_available": "Wall Availability",
        "basic_wall_functional": "Wall Functional",
    }
    out = df_basic[["EMIS Code"]].copy()
    for new_col, src_col in col_map.items():
        if src_col in df_basic.columns:
            out[new_col] = df_basic[src_col].apply(to_flag)
        else:
            out[new_col] = 0

    agg_cols = [c for c in out.columns if c != "EMIS Code"]
    return out.groupby("EMIS Code", dropna=False)[agg_cols].max().reset_index()


def load_and_merge_data(data_dir):
    print("Loading data...")
    df_infra = read_xlsx(data_dir, "Infrastructure.xlsx")
    df_furn = read_xlsx(data_dir, "Furniture (1).xlsx")
    df_teach = read_xlsx(data_dir, "Census TeacherInfo.xlsx")
    df_non_teach = read_xlsx(data_dir, "Census NonteacherInfo.xlsx")
    df_ptc = read_xlsx(data_dir, "Census Ptc Info.xlsx")

    df_basic = read_xlsx(data_dir, "Basic Facilities (1).xlsx")
    df_disabled = read_xlsx(data_dir, "Census Disabled Students.xlsx")
    df_itlab = read_xlsx(data_dir, "Census It Lab.xlsx")
    df_enroll = read_xlsx(data_dir, "Classwise Enrollment.xlsx")
    df_sanctioned_non_teach = read_xlsx(data_dir, "Sancioned Filled Non Teaching Staff.xlsx")
    df_sanctioned_teach = read_xlsx(data_dir, "Sanctioned Filled Teaching Staff.xlsx")

    print("Processing aggregate features...")
    teacher_agg = aggregate_teacher(df_teach)
    non_teacher_agg = aggregate_non_teacher(df_non_teach)
    ptc_agg = aggregate_ptc(df_ptc)
    basic_agg = aggregate_basic_facilities(df_basic)
    disabled_agg = aggregate_disabled(df_disabled)
    itlab_agg = aggregate_it_lab(df_itlab)
    enrollment_agg = aggregate_enrollment(df_enroll)
    sanctioned_nt_agg = aggregate_sanctioned_non_teaching(df_sanctioned_non_teach)
    sanctioned_t_agg = aggregate_sanctioned_teaching(df_sanctioned_teach)

    print("Merging datasets...")
    df_merged = pd.merge(df_infra, df_furn, on="EMIS Code", how="left", suffixes=("", "_furn"))

    for extra in [
        teacher_agg,
        non_teacher_agg,
        ptc_agg,
        basic_agg,
        disabled_agg,
        itlab_agg,
        enrollment_agg,
        sanctioned_nt_agg,
        sanctioned_t_agg,
    ]:
        df_merged = pd.merge(df_merged, extra, on="EMIS Code", how="left")

    print("Creating target label...")
    for col in [
        "Presentbalance",
        "Fund Recieved from Govt in This Year",
        "No of Class Room Need Reconstruction",
        "Desks Two Seater New Required",
        "Fans New Required",
        "total_teachers",
        "teach_vacancy_posts",
        "nonteach_vacancy_posts",
    ]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].apply(clean_num)

    df_merged["is_critical"] = (
        (df_merged.get("Whole Building Needs Reconstruction", "").astype(str).str.lower() == "yes")
        | (df_merged.get("No of Class Room Need Reconstruction", 0).fillna(0) > 0)
        | (df_merged.get("Desks Two Seater New Required", 0).fillna(0) > 0)
        | (df_merged.get("Fans New Required", 0).fillna(0) > 0)
        | (df_merged.get("total_teachers", 0).fillna(0) <= 1)
        | (df_merged.get("Presentbalance", 0).fillna(0) < 1000)
        | (df_merged.get("teach_vacancy_posts", 0).fillna(0) >= 3)
        | (df_merged.get("nonteach_vacancy_posts", 0).fillna(0) >= 2)
    ).astype(int)

    cols_to_drop = [c for c in df_merged.columns if c.endswith("_furn")]
    if cols_to_drop:
        df_merged.drop(columns=cols_to_drop, inplace=True)

    return df_merged


def preprocess_features(df):
    base_features = [
        "Gender",
        "LEVEL",
        "District",
        "BuildingOwnerShip",
        "TotalLand_Allocated",
        "Covered_Area",
        "UnCovered_Area",
        "No of Pakka Class Rooms",
        "No. Of Pakka Other Rooms",
        "Desks Two Seater Useable",
        "Tablet Chairs Useable",
        "Fans Useable",
        "total_teachers",
        "avg_bps_teachers",
        "qualified_teachers",
        "total_non_teachers",
        "Presentbalance",
        "Fund Recieved from Govt in This Year",
        "itlab_available",
        "itlab_functional",
        "itlab_total_computers",
        "itlab_functional_computers",
        "itlab_internet_available",
        "total_enrollment",
        "enrollment_primary",
        "enrollment_middle_secondary",
        "enrollment_higher_secondary",
        "teach_sanctioned_posts",
        "teach_filled_posts",
        "teach_vacancy_posts",
        "nonteach_sanctioned_posts",
        "nonteach_filled_posts",
        "nonteach_vacancy_posts",
        "disabled_students_total",
        "basic_electric_available",
        "basic_electric_functional",
        "basic_water_available",
        "basic_water_functional",
        "basic_toilet_available",
        "basic_toilet_functional",
        "basic_wall_available",
        "basic_wall_functional",
    ]

    for f in base_features:
        if f not in df.columns:
            df[f] = np.nan

    if "is_critical" in df.columns:
        return df[base_features], df["is_critical"]
    return df[base_features]


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.getenv("DATA_DIR", os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw")))

    df = load_and_merge_data(DATA_DIR)
    print(f"Final Data Shape: {df.shape}")
    print(f"Critical Schools: {int(df['is_critical'].sum())} out of {len(df)}")

    X, y = preprocess_features(df.copy())
    print(f"Features selected: {X.shape[1]}")

    out_path = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "processed_school_data.csv"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
