from pathlib import Path

from agent import run_agent
from data_processing import process_data
from detection import detect_anomalies
from explanation import explain_anomalies


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def run_pipeline(input_file=DATA_DIR / "sample_transactions.csv"):
    processed = DATA_DIR / "processed_transactions.csv"
    final = DATA_DIR / "final_transactions.csv"
    explained = DATA_DIR / "explained_transactions.csv"
    agent_output = DATA_DIR / "agent_output.csv"

    process_data(input_file, processed)
    detect_anomalies(processed, final)
    explain_anomalies(final, explained)
    return run_agent(explained, agent_output)


if __name__ == "__main__":
    print(run_pipeline())
