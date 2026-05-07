import pandas as pd


def assign_risk(row):
    if row["anomaly"] == 0:
        return "LOW"
    if abs(row["amount"]) > 500000 or row["vendor_frequency"] == 1:
        return "HIGH"
    return "MEDIUM"


def suggest_action(risk_level):
    if risk_level == "HIGH":
        return "Immediate audit review required"
    if risk_level == "MEDIUM":
        return "Verify vendor, category, and transaction evidence"
    return "No action needed"


def run_agent(input_path, output_path):
    df = pd.read_csv(input_path)
    df["risk_level"] = df.apply(assign_risk, axis=1)
    df["suggested_action"] = df["risk_level"].apply(suggest_action)
    df["agent_decision"] = df["risk_level"] + " - " + df["suggested_action"]
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_agent("../data/explained_transactions.csv", "../data/agent_output.csv")
    print(df)
