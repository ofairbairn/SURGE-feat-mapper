"""
Check equilibrium data for missingness
"""

import json
from pathlib import Path

import pandas as pd
from Mapper.preprocess import analyze_missingness

def check_missingness(file_path: str, output_dir: str = "runs/preprocess_experiment"):
    df = pd.read_csv(file_path)
    missing_summary = df.isnull().sum()
    print("Missing values summary:")
    print(missing_summary)

    report = analyze_missingness(df, output_dir="runs/preprocess_experiment", save_plots=True)
    plots_dir = Path(output_dir).resolve()
    for kind, name in report["plots"].items():
        print(f"Saved {kind} plot: {plots_dir / name}")
    for kind, error in report["plot_errors"].items():
        print(f"Failed to save {kind} plot: {error}")

    report_path = plots_dir / "data_preprocessing.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved missingness report: {report_path}")

if __name__ == "__main__":
    file_path = "C:\\Users\\Bipo1\\.cache\\huggingface\\hub\\datasets--SURGE-AIML--tokamakergen-nstxu-run10k\\snapshots\\a5830aeabf742f48dd4595e53376885f221be46f\\nstxu_RUN_10k_light\\nstxu_RUN_10k\\Equil_data_with_synthetic.csv"
    check_missingness(file_path)