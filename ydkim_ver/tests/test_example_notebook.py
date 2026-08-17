"""Structural and execution checks for the canonical example notebook.

AI disclosure: this test file was generated with OpenAI Codex assistance on
2026-08-17 and must not be marked verified until independently reviewed.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_qst_pipeline.ipynb"


def test_example_notebook_is_executed_without_errors():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    assert notebook["nbformat"] == 4
    assert len(code_cells) >= 6
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_example_notebook_contains_backend_and_visual_evidence():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ]
    stream_text = "".join(
        "".join(output.get("text", []))
        for output in outputs
        if output.get("output_type") == "stream"
    )
    embedded_types = {
        mime_type
        for output in outputs
        for mime_type in output.get("data", {})
    }

    assert "Counts exactly equal: True" in stream_text
    assert "linear physical=False" in stream_text
    assert "mle    physical=True" in stream_text
    assert "image/png" in embedded_types


def test_example_notebook_retains_ai_disclosure():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "AI disclosure" in markdown
    assert "Independent-review checklist" in markdown
