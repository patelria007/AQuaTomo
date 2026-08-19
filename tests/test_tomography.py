import tempfile
import unittest
import csv
import json
from pathlib import Path

import numpy as np

from nbqst.backend import scalar
from nbqst.denoise import depolarizing_shrinkage, low_rank_projection, project_density_matrix
from nbqst.io import load_measurement_bundle, save_measurement_bundle
from nbqst.cli import main as cli_main
from nbqst.experiment import benchmark, summarize
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
    NeuralTomographyModel,
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
from nbqst.pipeline import TomographyPipeline
from nbqst.shadows import (
    ClassicalShadowData,
    ClassicalShadowProtocol,
    PauliObservable,
    estimate_observable_from_measurements,
    observable_expectation,
)
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


class BenchmarkTimingTests(unittest.TestCase):
    @staticmethod
    def one_qubit_model():
        return NeuralTomographyModel(
            n_qubits=1,
            settings=complete_pauli_settings(1),
            weights=(np.zeros((6, 4)),),
            biases=(np.zeros(4),),
        )

    def test_synchronized_three_method_timing_records(self):
        records = benchmark(
            qubits=(1,),
            shots=(25,),
            state_types=("product",),
            states_per_case=1,
            methods=("li", "mle", "nn"),
            neural_models={1: self.one_qubit_model()},
            mle_iterations=2,
            warmup_rounds=0,
            timing_repeats=2,
            seed=41,
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(
            {row["method"] for row in records},
            {"linear_inversion", "maximum_likelihood", "neural_network"},
        )
        for row in records:
            self.assertGreaterEqual(row["reconstruction_seconds"], 0.0)
            self.assertGreaterEqual(row["fidelity_seconds"], 0.0)
            self.assertAlmostEqual(
                row["method_total_seconds"],
                row["reconstruction_seconds"] + row["fidelity_seconds"],
            )
            self.assertEqual(row["total_shots"], 75)
            self.assertIn("device_name", row)
        summary = summarize(records)
        self.assertEqual(len(summary), 3)
        self.assertTrue(all(row["timed_samples"] == 2 for row in summary))

    def test_cli_writes_backend_specific_timing_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_path = root / "model_1q.npz"
            save_neural_model(model_path, self.one_qubit_model())
            output = root / "timing.csv"
            cli_main(
                [
                    "benchmark",
                    "--qubits", "1",
                    "--shots", "20",
                    "--state-types", "haar",
                    "--states", "1",
                    "--methods", "li", "mle", "nn",
                    "--neural-model", f"1={model_path}",
                    "--mle-iterations", "2",
                    "--warmup-rounds", "0",
                    "--timing-repeats", "1",
                    "--output", str(output),
                ]
            )
            summary_path = root / "timing_summary.csv"
            manifest_path = root / "timing_manifest.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("fidelity_seconds", rows[0])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_count"], 3)
            self.assertTrue(manifest["timing_methodology"]["reconstruction_and_fidelity_timed_separately"])


class ClassicalShadowTests(unittest.TestCase):
    def test_pauli_observable_validation_and_exact_expectation(self):
        zero = np.diag([1.0, 0.0]).astype(complex)
        self.assertEqual(PauliObservable("z").label, "Z")
        self.assertEqual(PauliObservable("IXZ").weight, 2)
        self.assertAlmostEqual(observable_expectation(zero, "Z"), 1.0)
        self.assertAlmostEqual(observable_expectation(zero, "X"), 0.0)
        with self.assertRaises(ValueError):
            PauliObservable("AB")

    def test_shadow_estimator_formula_and_identity(self):
        shadow = ClassicalShadowData(
            n_qubits=1,
            basis_codes=np.asarray([[0], [1], [2]], dtype=np.int8),
            outcomes=np.asarray([[0], [0], [0]], dtype=np.int8),
        )
        protocol = ClassicalShadowProtocol()
        self.assertAlmostEqual(protocol.estimate(shadow, "Z").value, 1.0)
        self.assertAlmostEqual(protocol.estimate(shadow, PauliObservable("I", 2.5)).value, 2.5)

    def test_acquisition_is_unbiased_for_known_z_state(self):
        zero = np.diag([1.0, 0.0]).astype(complex)
        protocol = ClassicalShadowProtocol(median_of_means_groups=5)
        shadow = protocol.acquire(zero, 12_000, rng=51)
        estimates = {item.observable.label: item for item in protocol.estimate_many(shadow, ("X", "Y", "Z"))}
        self.assertLess(abs(estimates["X"].value), 0.06)
        self.assertLess(abs(estimates["Y"].value), 0.06)
        self.assertLess(abs(estimates["Z"].value - 1.0), 0.06)
        self.assertEqual(estimates["Z"].samples, 12_000)

    def test_direct_measurement_observable_estimate(self):
        truth = ghz_state(2)
        data = exact_pauli_measurements(truth)
        estimate = estimate_observable_from_measurements(data, "ZZ")
        self.assertAlmostEqual(estimate.value, 1.0)
        self.assertEqual(estimate.method, "direct_pauli_setting")

    def test_object_oriented_pipeline(self):
        pipeline = TomographyPipeline(seed=61, mle_iterations=2)
        truth, data, results = pipeline.run(
            n_qubits=1,
            shots_per_setting=200,
            state_type="product",
            methods=("projected_li", "mle"),
        )
        self.assertEqual(truth.shape, (2, 2))
        self.assertEqual(data.n_qubits, 1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.is_physical for result in results))


if __name__ == "__main__":
    unittest.main()
