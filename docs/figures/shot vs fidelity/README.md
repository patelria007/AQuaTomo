# 3-qubit shot count versus reconstruction fidelity

This verification compares random product-pure, Haar-random pure, and
full-rank Ginibre mixed states without readout error. Each point is the mean of
100 independently generated 3-qubit target states. For each target, all 27
local-Pauli settings are sampled with multinomial shot noise, reconstructed by
Pauli linear inversion, and projected to the nearest positive-semidefinite,
unit-trace density matrix. Fidelity is squared Uhlmann fidelity.

The x-axis is **shots per Pauli setting**, consistent with the rest of this
repository. Therefore a value of `N` on the x-axis corresponds to `27 × N`
total measurement shots for a 3-qubit experiment.

Run from the repository root:

```powershell
python "verification/shot vs fidelity/shot_vs_fidelity.py"
```

The generated PNG shows all three ensemble-mean curves with 95% confidence
intervals and a horizontal 99% fidelity target. The CSV contains the plotted
summary values, while the JSON also retains every trial and the first tested
shot count whose ensemble mean reaches 99%.

For symmetric 99% readout fidelity without readout-error mitigation, run:

```powershell
python "verification/shot vs fidelity/shot_vs_fidelity.py" `
  --readout-fidelity 0.99 `
  --output "verification/shot vs fidelity/shot_vs_fidelity_readout_99.png" `
  --csv "verification/shot vs fidelity/shot_vs_fidelity_readout_99_results.csv" `
  --json "verification/shot vs fidelity/shot_vs_fidelity_readout_99_results.json"
```

Here the simulator applies `P(0|0) = P(1|1) = 0.99` independently to each
qubit before multinomial sampling. Reconstruction is deliberately uncorrected,
so the result exposes the systematic fidelity ceiling caused by readout error.

For asymmetric readout with `P(0|0) = 1.00` and `P(1|1) = 0.98`, run:

```powershell
python "verification/shot vs fidelity/shot_vs_fidelity.py" `
  --readout-fidelity-0 1.00 `
  --readout-fidelity-1 0.98 `
  --output "verification/shot vs fidelity/shot_vs_fidelity_readout_f0_100_f1_98.png" `
  --csv "verification/shot vs fidelity/shot_vs_fidelity_readout_f0_100_f1_98_results.csv" `
  --json "verification/shot vs fidelity/shot_vs_fidelity_readout_f0_100_f1_98_results.json"
```

The focused no-readout product/pure result uses 500 target states and a denser
shot grid around the 99% crossing:

```powershell
python "verification/shot vs fidelity/shot_vs_fidelity.py" `
  --families product pure `
  --trials 500 `
  --shot-counts 100 150 220 330 470 680 1000 1500 2200 3000 3500 4000 4250 4500 4750 5000 5250 5500 5750 6000 6250 6500 6750 7000 7250 7500 7750 8000 8500 9000 10000 `
  --output "verification/shot vs fidelity/shot_vs_fidelity_product_pure_refined_no_readout.png" `
  --csv "verification/shot vs fidelity/shot_vs_fidelity_product_pure_refined_no_readout_results.csv" `
  --json "verification/shot vs fidelity/shot_vs_fidelity_product_pure_refined_no_readout_results.json"
```

A sparse six-qubit package-baseline run (the unmodified package measurement and
reconstruction kernels, 10 trials per condition) is produced with:

```powershell
python "verification/shot vs fidelity/shot_vs_fidelity.py" `
  --qubits 6 `
  --families product pure `
  --trials 10 `
  --shot-counts 100 300 1000 3000 10000 `
  --output "verification/shot vs fidelity/shot_vs_fidelity_6q_product_pure_sparse_no_readout.png" `
  --csv "verification/shot vs fidelity/shot_vs_fidelity_6q_product_pure_sparse_no_readout_results.csv" `
  --json "verification/shot vs fidelity/shot_vs_fidelity_6q_product_pure_sparse_no_readout_results.json"
```
