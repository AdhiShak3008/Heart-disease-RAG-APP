from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "data" / "processed" / "dataset_index.csv"

OUTPUT = BASE_DIR / "data" / "processed"

RANDOM_STATE = 42


def main():

    df = pd.read_csv(DATASET)

    patient_labels = df.groupby("patient_id")["murmur"].first().reset_index()

    train_patients, test_patients = train_test_split(
        patient_labels,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=patient_labels["murmur"],
    )

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=train_patients["murmur"],
    )

    train_df = df[df["patient_id"].isin(train_patients["patient_id"])]

    val_df = df[df["patient_id"].isin(val_patients["patient_id"])]

    test_df = df[df["patient_id"].isin(test_patients["patient_id"])]

    train_df.to_csv(
        OUTPUT / "train.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT / "val.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT / "test.csv",
        index=False,
    )

    print("=" * 50)

    print("Patients")

    print(f"Train      : {len(train_patients)}")
    print(f"Validation : {len(val_patients)}")
    print(f"Test       : {len(test_patients)}")

    print()

    print("Samples")

    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("=" * 50)


if __name__ == "__main__":
    main()
