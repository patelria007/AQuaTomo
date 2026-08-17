"""Portable serialization helpers; NumPy is intentionally the file format layer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .backend import to_numpy
from .measurements import MeasurementData


def write_csv(path, records):
    records = list(records)
    if not records:
        raise ValueError("No records to write")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_measurement_bundle(path, states, datasets, metadata=None):
    """Save states and counts in a non-pickled NPZ plus JSON metadata."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"states": np.stack([to_numpy(state) for state in states])}
    manifest = {"datasets": [], "metadata": metadata or {}}
    for i, data in enumerate(datasets):
        item = {
            "n_qubits": data.n_qubits,
            "shots_per_setting": data.shots_per_setting,
            "settings": list(data.settings),
        }
        for setting, counts in data.counts.items():
            payload[f"counts_{i}_{setting}"] = to_numpy(counts)
        manifest["datasets"].append(item)
    payload["manifest_json"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(target, **payload)


def load_measurement_bundle(path, *, xp=np):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        states = [xp.asarray(state) for state in archive["states"]]
        datasets = []
        for i, item in enumerate(manifest["datasets"]):
            counts = {setting: xp.asarray(archive[f"counts_{i}_{setting}"]) for setting in item["settings"]}
            datasets.append(MeasurementData(item["n_qubits"], counts, item["shots_per_setting"]))
    return states, datasets, manifest["metadata"]

