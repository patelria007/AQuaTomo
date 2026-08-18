import tempfile
import unittest
from pathlib import Path

import numpy as np

from nbqst.backend import scalar
from nbqst.cli import main as cli_main
from nbqst.denoise import depolarizing_shrinkage, low_rank_projection, project_density_matrix
from nbqst.io import load_measurement_bundle, save_measurement_bundle
from nbqst.measurements import (
    MeasurementData,
    _apply_readout_fidelity,
    complete_pauli_settings,
    exact_pauli_measurements,
    global_pauli_settings,
    simulate_pauli_measurements,
    split_measurement_data,
)
from nbqst.metrics import fidelity, minimum_eigenvalue, purity
from nbqst.noise import global_depolarizing_channel, local_depolarizing_channel
from nbqst.reconstruction import factorized_mle, linear_inversion_pauli
from nbqst.states import ghz_state, haar_random_pure, random_mixed_state, random_product_state


class StateTests(unittest.TestCase):
    def assert_density(self, rho, tolerance=1e-10):
        self.assertTrue(np.allclose(rho, rho.conj().T, atol=tolerance))
        self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=10)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(rho).min()), -tolerance)

    def test_state_generators(self):
        for generator in (random_product_state, haar_random_pure, random_mixed_state):
            self.assert_density(generator(2, rng=3))

    def test_pure_and_mixed_purity(self):
        self.assertAlmostEqual(scalar(purity(haar_random_pure(2, rng=2))), 1.0, places=10)
        self.assertLess(scalar(purity(random_mixed_state(2, rng=2))), 1.0)


class MeasurementTests(unittest.TestCase):
    def test_complete_settings(self):
        self.assertEqual(len(complete_pauli_settings(3)), 27)
        self.assertEqual(len(global_pauli_settings(3)), 3)

    def test_multinomial_counts(self):
        rho = random_product_state(2, rng=1)
        data = simulate_pauli_measurements(rho, 123, rng=4)
        self.assertTrue(data.informationally_complete)
        self.assertTrue(all(int(np.sum(c)) == 123 for c in data.counts.values()))

    def test_perfect_readout_preserves_seeded_counts(self):
        rho = random_product_state(2, rng=1)
        ideal = simulate_pauli_measurements(rho, 123, rng=4)
        perfect = simulate_pauli_measurements(
            rho, 123, rng=4, readout_fidelity_0=1.0, readout_fidelity_1=1.0
        )
        for setting in ideal.settings:
            self.assertTrue(np.array_equal(ideal.counts[setting], perfect.counts[setting]))

    def test_asymmetric_readout_probabilities(self):
        self.assertTrue(
            np.allclose(_apply_readout_fidelity([1.0, 0.0], 0.8, 0.9), [0.8, 0.2])
        )
        self.assertTrue(
            np.allclose(_apply_readout_fidelity([0.0, 1.0], 0.8, 0.9), [0.1, 0.9])
        )

    def test_readout_big_endian_ordering(self):
        observed = _apply_readout_fidelity(
            [0.0, 0.0, 0.0, 1.0], [0.9, 0.8], [0.8, 0.6]
        )
        self.assertTrue(np.allclose(observed, [0.08, 0.12, 0.32, 0.48]))

    def test_noisy_readout_conserves_shots(self):
        rho = random_product_state(2, rng=1)
        data = simulate_pauli_measurements(
            rho,
            123,
            rng=4,
            readout_fidelity_0=[0.98, 0.97],
            readout_fidelity_1=[0.96, 0.95],
        )
        self.assertTrue(all(int(np.sum(c)) == 123 for c in data.counts.values()))

    def test_invalid_readout_fidelity_rejected(self):
        rho = random_product_state(2, rng=1)
        invalid = (
            {"readout_fidelity_0": 0.9},
            {"readout_fidelity_0": [0.9, 0.9], "readout_fidelity_1": [0.8, 0.8, 0.8]},
            {"readout_fidelity_0": [0.9, 1.1], "readout_fidelity_1": 0.9},
            {"readout_fidelity_0": np.nan, "readout_fidelity_1": 0.9},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_pauli_measurements(rho, 10, **options)

    def test_train_validation_split_conserves_counts(self):
        rho = random_product_state(2, rng=1)
        data = simulate_pauli_measurements(rho, 101, rng=4)
        train, validation = split_measurement_data(data, 0.2, rng=5)
        self.assertEqual(train.shots_per_setting, 81)
        self.assertEqual(validation.shots_per_setting, 20)
        for setting in data.settings:
            self.assertTrue(np.array_equal(train.counts[setting] + validation.counts[setting], data.counts[setting]))

    def test_exact_inversion_one_and_two_qubits(self):
        for rho in (haar_random_pure(1, rng=8), ghz_state(2)):
            estimate = linear_inversion_pauli(exact_pauli_measurements(rho))
            self.assertTrue(np.allclose(rho, estimate, atol=1e-10))

    def test_incomplete_data_rejected(self):
        rho = ghz_state(2)
        full = exact_pauli_measurements(rho)
        counts = {k: full.counts[k] for k in global_pauli_settings(2)}
        with self.assertRaises(ValueError):
            linear_inversion_pauli(MeasurementData(2, counts, full.shots_per_setting))


class ReconstructionTests(unittest.TestCase):
    def test_projection_is_physical(self):
        bad = np.array([[1.2, 0.4], [0.4, -0.2]], dtype=complex)
        estimate = project_density_matrix(bad)
        self.assertGreaterEqual(scalar(minimum_eigenvalue(estimate)), -1e-12)
        self.assertAlmostEqual(float(np.trace(estimate).real), 1.0, places=10)

    def test_low_rank_and_shrinkage(self):
        rho = random_mixed_state(2, rng=9)
        low = low_rank_projection(rho, 1)
        shrunk = depolarizing_shrinkage(rho, 0.5)
        self.assertGreaterEqual(scalar(minimum_eigenvalue(low)), -1e-12)
        self.assertGreaterEqual(scalar(minimum_eigenvalue(shrunk)), -1e-12)
        self.assertAlmostEqual(float(np.trace(low).real), 1.0, places=10)

    def test_factorized_mle_is_physical_and_monotone(self):
        truth = haar_random_pure(1, rng=20)
        data = simulate_pauli_measurements(truth, 300, rng=21)
        initial = project_density_matrix(linear_inversion_pauli(data))
        estimate, history = factorized_mle(data, initial=initial, max_iter=50, return_history=True)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(history, history[1:])))
        self.assertGreaterEqual(scalar(minimum_eigenvalue(estimate)), -1e-12)
        self.assertGreater(scalar(fidelity(truth, estimate)), 0.8)


class NoiseAndIOTests(unittest.TestCase):
    def test_channels_preserve_density(self):
        rho = ghz_state(2)
        for noisy in (global_depolarizing_channel(rho, 0.2), local_depolarizing_channel(rho, 0.2)):
            self.assertAlmostEqual(float(np.trace(noisy).real), 1.0, places=10)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(noisy).min()), -1e-12)

    def test_bundle_round_trip(self):
        rho = random_product_state(1, rng=4)
        data = simulate_pauli_measurements(rho, 50, rng=5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle.npz"
            save_measurement_bundle(path, [rho], [data], {"purpose": "test"})
            states, datasets, metadata = load_measurement_bundle(path)
        self.assertTrue(np.allclose(states[0], rho))
        self.assertEqual(metadata["purpose"], "test")
        self.assertEqual(datasets[0].shots_per_setting, 50)

    def test_cli_readout_metadata_and_paired_options(self):
        with self.assertRaises(SystemExit):
            cli_main(["generate", "--readout-fidelity-0", "0.9"])

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "readout_bundle.npz"
            cli_main(
                [
                    "generate",
                    "--qubits",
                    "1",
                    "--samples",
                    "1",
                    "--shots",
                    "10",
                    "--readout-fidelity-0",
                    "0.9",
                    "--readout-fidelity-1",
                    "0.8",
                    "--output",
                    str(path),
                ]
            )
            _, _, metadata = load_measurement_bundle(path)
        self.assertEqual(metadata["readout_fidelity_0"], [0.9])
        self.assertEqual(metadata["readout_fidelity_1"], [0.8])


if __name__ == "__main__":
    unittest.main()
