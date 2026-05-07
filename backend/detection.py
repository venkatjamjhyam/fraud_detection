import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(file_path, output_path):
    df = pd.read_csv(file_path)
    if df.empty:
        df["anomaly"] = []
        df.to_csv(output_path, index=False)
        return df

    features = df[["amount_abs", "deviation", "vendor_frequency", "category_frequency"]].fillna(0)
    contamination = min(0.25, max(0.05, 6 / max(len(df), 1)))

    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(features)
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = detect_anomalies("../data/processed_transactions.csv", "../data/final_transactions.csv")
    print(df)
