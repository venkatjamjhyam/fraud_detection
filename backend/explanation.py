import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


def generate_explanation(row, avg_amount):
    reasons = []

    if abs(row["amount"]) > abs(avg_amount) * 3:
        reasons.append("Amount is significantly higher than average")
    if row["vendor_frequency"] == 1:
        reasons.append("New or rare vendor")
    if row.get("category_frequency", 2) == 1:
        reasons.append("Rare category or transaction purpose")
    if abs(row["amount"]) < abs(avg_amount) * 0.3:
        reasons.append("Unusually low transaction amount")

    if not reasons:
        return "ML model flagged this based on amount, vendor, and category behavior"
    return "; ".join(reasons)


def explain_anomalies(input_path, output_path):
    df = pd.read_csv(input_path)
    avg_amount = df["amount"].mean() if not df.empty else 0
    df["explanation"] = df.apply(
        lambda row: generate_explanation(row, avg_amount) if row["anomaly"] == 1 else "Normal",
        axis=1,
    )
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = explain_anomalies("../data/final_transactions.csv", "../data/explained_transactions.csv")
    print(df)
