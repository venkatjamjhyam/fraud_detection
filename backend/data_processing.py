import pandas as pd

from schema_detection import detect_schema, normalize_transactions


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.astype(str).str.strip()
    print("CSV Columns:", df.columns.tolist())
    return df


def clean_data(df, mapping=None, use_claude=True):
    df, _schema = normalize_transactions(df, mapping=mapping, use_claude=use_claude)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def create_features(df):
    avg_amount = df["amount"].mean()
    df["deviation"] = df["amount"] - avg_amount
    df["amount_abs"] = df["amount"].abs()

    vendor_freq = df["vendor"].value_counts()
    df["vendor_frequency"] = df["vendor"].map(vendor_freq)

    category_freq = df["category"].value_counts()
    df["category_frequency"] = df["category"].map(category_freq)
    return df


def inspect_schema(input_path, use_claude=True):
    df = load_data(input_path)
    return detect_schema(df, use_claude=use_claude)


def process_data(input_path, output_path, mapping=None, use_claude=True):
    df = load_data(input_path)
    df = clean_data(df, mapping=mapping, use_claude=use_claude)
    df = create_features(df)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    process_data("../data/sample_transactions.csv", "../data/processed_transactions.csv")
