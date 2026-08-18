"""Probe one dense tomography size under an explicit wall-time cap."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from nbqst.denoise import project_density_matrix
from nbqst.measurements import simulate_pauli_measurements
from nbqst.metrics import fidelity
from nbqst.reconstruction import linear_inversion_pauli
from nbqst.states import haar_random_pure


class WallTimeLimit(TimeoutError):
    pass


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "n_qubits": args.qubits,
        "shots_per_setting": args.shots,
        "settings": 3**args.qubits,
        "frequencies": 6**args.qubits,
        "pauli_coefficients": 4**args.qubits,
        "wall_time_cap_seconds": args.timeout,
        "status": "started",
    }

    def save():
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def timeout_handler(_signal, _frame):
        raise WallTimeLimit(f"wall-time cap of {args.timeout} seconds reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    started = time.perf_counter()
    try:
        rng = np.random.default_rng(args.seed)
        truth = haar_random_pure(args.qubits, rng=rng)
        stage = time.perf_counter()
        data = simulate_pauli_measurements(truth, args.shots, rng=rng)
        result["measurement_simulation_seconds"] = time.perf_counter() - stage
        result["status"] = "measurement_complete"
        save()
        stage = time.perf_counter()
        estimate = linear_inversion_pauli(data)
        result["linear_inversion_seconds"] = time.perf_counter() - stage
        result["projected_fidelity"] = float(fidelity(truth, project_density_matrix(estimate)))
        result["status"] = "complete"
    except WallTimeLimit as error:
        result["status"] = "wall_time_cap_reached"
        result["error"] = str(error)
    finally:
        signal.alarm(0)
        result["total_seconds"] = time.perf_counter() - started
        save()


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--qubits", type=int, required=True)
    result.add_argument("--shots", type=int, default=200)
    result.add_argument("--timeout", type=int, default=90)
    result.add_argument("--seed", type=int, default=20260819)
    result.add_argument("--output", default="results/verification_study/scaling_limit_probe.json")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
