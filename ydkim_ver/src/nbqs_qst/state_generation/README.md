# State generation

Hardware-agnostic target-state generation for the tomography pipeline is
implemented in one core file, [`state_generation.py`](state_generation.py).

```python
import numpy as np

from nbqs_qst.state_generation import random_haar_state

target = random_haar_state(np.asarray(0.0), n=3, seed=17)
rho = target.rho
```

Available families:

- independent local-Haar product pure states;
- global Haar-random pure states;
- induced Ginibre/Wishart random states with configurable `K`;
- Haar- or product-based states with exact target purity;
- deterministic GHZ and W references.

The package also exports purity, squared Uhlmann fidelity, and pure-state
overlap. The core uses `array_namespace(like)`, `complex128`, functional array
operations, and only stdlib `random.Random`.

Documentation:

- [`state_generation.md`](state_generation.md): function-level methods, API,
  examples, limitations, and validation commands;
- [`theory_notes.md`](theory_notes.md): literature review and design rationale;
- [`../../../docs/figures/state_generation/README.md`](../../../docs/figures/state_generation/README.md):
  poster figure methodology and outputs.

Validation commands:

```bash
python -m pytest -q tests/test_state_generation.py
python -m pytest -q tests/test_measurement.py
python tests/analysis/generate_state_generation_figures.py
```

AI disclosure: this organizational text was generated with AI assistance on
2026-08-17 and has not yet been independently verified.
