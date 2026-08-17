# Poster plot set

> **AI disclosure:** The plotting code, figure design, and this guide were
> created with AI assistance on 2026-08-17. Independently review the results
> before marking them verified or using them in the final submission.

Regenerate every figure from the project root:

```powershell
python measurement_generation\measure_test\generate_poster_plots.py
```

Each figure is exported as a 300-DPI PNG and a vector PDF. Exact plotted values,
fixed seeds, runtime results, and environment metadata are recorded in
`poster_plot_data.json`.

## Recommended poster use

- `poster_pauli_validation`: physical correctness and the `Y`-sign regression.
- `poster_bell_measurements`: Born probabilities versus finite-shot outcomes.
- `poster_shot_scaling`: statistical validation of the `1/sqrt(N)` law.
- `poster_five_expectation_shot_clouds`: finite-shot scatter around the five
  true values `+1`, `+0.5`, `0`, `-0.5`, and `-1`.
- `poster_three_expectation_shot_clouds`: compact version using only `+1`, `0`,
  and `-1`.
- `poster_aggregation_gain`: benefit of pooling every compatible setting.
- `poster_backend_comparison`: NumPy/JAX agreement and CPU runtime comparison.
- `poster_resource_scaling`: the exponential limit of exhaustive Pauli QST.

The strongest compact three-figure story is `poster_bell_measurements`,
`poster_shot_scaling`, and `poster_backend_comparison`. Add
`poster_aggregation_gain` when explaining the estimator design.

## Interpretation cautions

- Backend timings were measured locally on CPU in eager mode. They are not GPU
  benchmarks and should not be generalized to other machines.
- Runtime values can change between executions; the accuracy and fixed-seed
  comparisons should remain stable.
- The resource plot reports theoretical dense-array storage, not observed peak
  process memory, which also includes temporaries and framework overhead.
- Statistical plots use deterministic seed sets for reproducibility, not
  backend random-number generators.
