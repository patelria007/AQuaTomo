import tempfile
import unittest
from pathlib import Path

import numpy as np

from nbqst.backend import scalar
from nbqst.denoise import depolarizing_shrinkage, low_rank_projection, project_density_matrix
from nbqst.io import load_measurement_bundle, save_measurement_bundle
from nbqst.measurements import (
    MeasurementData,
    apply_readout_confusion,
    complete_pauli_settings,
    exact_pauli_measurements,
    global_pauli_settings,
    simulate_pauli_measurements,
    split_measurement_data,
)
from nbqst.metrics import fidelity, minimum_eigenvalue, purity
from nbqst.neural import (
    load_neural_model,
    neural_state_reconstruction,
    save_neural_model,
    train_neural_reconstructor,
)
from nbqst.noise import (
    amplitude_damping_channel,
    asymmetric_pauli_channel,
    coherent_rotation_channel,
    global_depolarizing_channel,
    local_depolarizing_channel,
    phase_damping_channel,
)
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

    def test_neural_cholesky_reconstruction_is_physical_and_serializable(self):
        rng = np.random.default_rng(31)
        states = [random_mixed_state(1, rng=rng) for _ in range(40)]
        datasets = [simulate_pauli_measurements(state, 300, rng=rng) for state in states]
        model, history = train_neural_reconstructor(
            states,
            datasets,
            hidden_layers=(24,),
            epochs=100,
            batch_size=10,
            validation_fraction=0.25,
            patience=25,
            seed=32,
            return_history=True,
        )
        estimate = neural_state_reconstruction(datasets[0], model)
        self.assertGreaterEqual(scalar(minimum_eigenvalue(estimate)), -1e-12)
        self.assertAlmostEqual(float(np.trace(estimate).real), 1.0, places=10)
        self.assertLess(min(row["validation_loss"] for row in history), history[0]["validation_loss"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            save_neural_model(path, model)
            restored = load_neural_model(path)
        self.assertTrue(np.allclose(estimate, neural_state_reconstruction(datasets[0], restored)))


class NoiseAndIOTests(unittest.TestCase):
    def test_channels_preserve_density(self):
        rho = ghz_state(2)
        channels = (
            global_depolarizing_channel(rho, 0.2),
            local_depolarizing_channel(rho, 0.2),
            amplitude_damping_channel(rho, 0.2),
            phase_damping_channel(rho, 0.2),
            asymmetric_pauli_channel(rho, p_x=0.03, p_y=0.02, p_z=0.08),
            coherent_rotation_channel(rho, 0.13, axis="Y"),
        )
        for noisy in channels:
            self.assertAlmostEqual(float(np.trace(noisy).real), 1.0, places=10)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(noisy).min()), -1e-12)

    def test_noise_channels_match_analytic_one_qubit_cases(self):
        excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
        damped = amplitude_damping_channel(excited, 0.3)
        self.assertTrue(np.allclose(damped, np.diag([0.3, 0.7])))

        plus = np.full((2, 2), 0.5, dtype=complex)
        dephased = phase_damping_channel(plus, 0.25)
        self.assertAlmostEqual(float(dephased[0, 1].real), 0.375, places=12)

        rotated = coherent_rotation_channel(plus, 0.7, axis="Z")
        self.assertAlmostEqual(float(np.trace(rotated @ rotated).real), 1.0, places=12)

    def test_global_depolarizing_shrinks_bloch_vector(self):
        plus = np.full((2, 2), 0.5, dtype=complex)
        noisy = global_depolarizing_channel(plus, 0.2)
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.assertAlmostEqual(float(np.trace(noisy @ sigma_x).real), 0.8, places=12)

    def test_readout_confusion_is_classical_and_analytic(self):
        confusion = np.array([[0.9, 0.2], [0.1, 0.8]])
        observed = apply_readout_confusion(np.array([1.0, 0.0]), 1, confusion)
        self.assertTrue(np.allclose(observed, [0.9, 0.1]))
        observed_two = apply_readout_confusion(np.array([1.0, 0.0, 0.0, 0.0]), 2, confusion)
        self.assertTrue(np.allclose(observed_two, [0.81, 0.09, 0.09, 0.01]))

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


if __name__ == "__main__":
    unittest.main()
