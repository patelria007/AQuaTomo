# `pipeline.py` — Complete tomography workflow

`run_tomography` connects the three project stages without hiding state
generation. A caller first creates a `GeneratedState`, then supplies it to the
pipeline with a shot count, measurement seed, and reconstruction method.

```python
import numpy as np
import nbqs_qst as qst

target = qst.random_haar_state(np.asarray(0.0), 2, seed=17)
run = qst.run_tomography(
    target,
    512,
    measurement_seed=23,
    method="mle",
    initial="pls",
)
print(run.summary())
```

The returned frozen `TomographyRun` contains the target, raw measurement
dataset, reconstruction diagnostics, purity values, and fidelity. Fidelity is
`None` when the estimate is nonphysical, which commonly occurs for finite-shot
linear inversion. This prevents the Uhlmann density-matrix metric from being
applied outside its mathematical domain.

The function introduces no new randomness: measurement sampling still uses the
single stdlib `random.Random` path in `pauli_measurement.py`. Arrays remain on
the target backend throughout the core pipeline.

AI disclosure: this API, example, and companion text were generated with
OpenAI Codex assistance on 2026-08-17. Independent review is pending.
