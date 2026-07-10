"""Prepare PNW seismology sequence dataset and run SURGE: CNN1D + VAE."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

import surge  # noqa: F401 - trigger model registration
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

WAVEFORMS_H5 = Path(r"C:\Users\Bipo1\Downloads\PNW Seismic Test Data\microPNW_waveforms.hdf5")
METADATA_CSV = Path(r"C:\Users\Bipo1\Downloads\PNW Seismic Test Data\microPNW_metadata.csv")

CSV_OUT = Path("data/datasets/pnw_seismic/pnw_seq2seq.csv")
SPEC_PATH = Path("examples/configs/pnw_cnn1d_vae.yaml")

WINDOW_SAMPLES = 1024
INPUT_CHANNEL = 0
OUTPUT_CHANNEL = 2


def _parse_trace_name(trace_name: str) -> tuple[str, int]:
    # Example: bucket1$0,:3,:15001
    left = trace_name.split(",", 1)[0]
    bucket, row = left.split("$", 1)
    return bucket, int(row)


def _extract_window(signal: np.ndarray, center: int, window: int) -> np.ndarray:
    half = window // 2
    start = center - half
    end = start + window

    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0

    if end > signal.shape[0]:
        pad_right = end - signal.shape[0]
        end = signal.shape[0]
    else:
        pad_right = 0

    clip = signal[start:end]
    if pad_left or pad_right:
        clip = np.pad(clip, (pad_left, pad_right), mode="constant")
    return clip.astype(np.float32)


def prepare_csv(output_path: Path = CSV_OUT) -> Path:
    meta = pd.read_csv(METADATA_CSV)
    rows = []

    with h5py.File(WAVEFORMS_H5, "r") as h5:
        for _, rec in meta.iterrows():
            trace_name = str(rec["trace_name"])
            bucket, row_idx = _parse_trace_name(trace_name)
            waveform = np.asarray(h5["data"][bucket][row_idx], dtype=np.float32)  # (3, n_samples)

            p_arrival = rec.get("trace_P_arrival_sample")
            if pd.isna(p_arrival):
                center = waveform.shape[1] // 2
            else:
                center = int(float(p_arrival))

            x_sig = _extract_window(waveform[INPUT_CHANNEL], center, WINDOW_SAMPLES)
            y_sig = _extract_window(waveform[OUTPUT_CHANNEL], center, WINDOW_SAMPLES)

            row = {f"input_t{i:04d}": float(v) for i, v in enumerate(x_sig)}
            row.update({f"output_t{i:04d}": float(v) for i, v in enumerate(y_sig)})
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote dataset: {output_path} shape={df.shape}")
    return output_path


def run_spec(spec_path: Path = SPEC_PATH) -> dict:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec = SurrogateWorkflowSpec.from_dict(payload)
    return run_surrogate_workflow(spec, invocation={"script": str(Path(__file__).resolve())})


def main() -> None:
    prepare_csv()
    summary = run_spec()
    print("Workflow complete.")
    for model in summary.get("models", []):
        print(f"{model.get('name', model.get('key'))}: {model.get('metrics', {})}")

    run_root = summary.get("artifacts", {}).get("root")
    if run_root:
        viz_result = viz_run(Path(run_root))
        print("Visualization complete.")
        for path in viz_result.get("saved_paths", []):
            print(path)


if __name__ == "__main__":
    main()
