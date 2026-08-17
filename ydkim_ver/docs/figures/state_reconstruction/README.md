# State-reconstruction poster figures

> **AI disclosure:** The plotting code, figure design, and this guide were
> generated with AI assistance on 2026-08-17. Independently review the data
> and interpretation before using the figures or marking them verified.

Regenerate the figures from the project root:

```powershell
python state_reconstruction\reconstruction_test\generate_poster_plots.py
```

Every figure is saved as a 300-DPI PNG and vector PDF. Exact plotted values,
fixed seeds, convergence flags, runtimes, and environment metadata are stored
in `poster_plot_data.json`.

## Recommended poster story

- `poster_reconstruction_quality`: PLS/MLE infidelity versus shots for
  one-qubit `|+y>`, two-qubit Bell, and four-qubit GHZ states, plus the
  fraction of nonphysical linear-inversion estimates.
- `poster_mle_convergence`: monotone accepted multinomial NLL and the true
  versus reconstructed Bell-state density-matrix magnitude.
- `poster_reconstruction_backends`: fixed-count NumPy/JAX agreement and local
  eager-CPU wall time.
- `poster_reconstruction_resource_limit`: why dense full QST cannot satisfy a
  literal 20-qubit benchmark without a structured or matrix-free extension.

The strongest compact two-figure result is `poster_reconstruction_quality`
plus `poster_mle_convergence`. Add the backend panel to demonstrate the Array
API objective and the resource panel to state the scaling boundary honestly.

## Interpretation cautions

- Shaded fidelity bands are the 16th–84th empirical percentiles over 20 fixed
  Monte Carlo seeds, not formal confidence intervals.
- Fidelity is reported only for physical PLS/MLE states. Linear inversion is
  summarized by its nonphysical rate instead of clipping it before evaluation.
- The backend plot is a local CPU/eager measurement after one warm-up run. It
  is not a GPU benchmark and must not be generalized to other hardware.
- The resource curve is theoretical matrix storage and excludes temporaries,
  framework overhead, measurement arrays, and optimizer state.
