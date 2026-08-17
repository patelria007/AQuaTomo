# Executable examples

`complete_qst_pipeline.ipynb` is the canonical end-to-end package example. It
uses only the installed `nbqs_qst` public API and does not depend on any other
repository or notebook.

Install the example dependencies and execute it from the project root:

```powershell
python -m pip install -e ".[example,test]"
python -m nbconvert --to notebook --execute --inplace examples/complete_qst_pipeline.ipynb
```

The committed notebook contains outputs from a fixed-seed NumPy/JAX CPU run.
Re-execution can change timing values but should preserve counts and numerical
agreement within the stated tolerances.

AI disclosure: the notebook design, code, narrative, and this guide were
generated with OpenAI Codex assistance on 2026-08-17. Independent review is
pending.
