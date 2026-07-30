import pandas as pd

files = [
    "backend/data/processed/train.csv",
    "backend/data/processed/val.csv",
    "backend/data/processed/test.csv",
]

for fname in files:
    df = pd.read_csv(fname)
    print(fname)
    print("rows", len(df))
    if "patient_id" in df.columns and "murmur" in df.columns:
        counts = df.groupby("patient_id")["murmur"].nunique()
        print("unique patient count", df["patient_id"].nunique())
        print("patient label unique counts (value counts of nunique):")
        print(counts.value_counts().sort_index().to_string())
        print("label distribution patient-level:")
        print(df.groupby("patient_id")["murmur"].first().value_counts().to_string())
        print("label distribution sample-level:")
        print(df["murmur"].value_counts().to_string())
        print("average recordings per patient by class:")
        recordings_per_patient = (
            df.groupby(["patient_id", "murmur"]).size().groupby(level=1).mean()
        )
        print(recordings_per_patient.to_string())
    print("---")
