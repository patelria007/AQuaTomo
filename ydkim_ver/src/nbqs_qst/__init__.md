# Unified `nbqs_qst` package API

The `nbqs_qst` namespace re-exports the stable public functions and immutable
records from state generation, measurement generation, and state
reconstruction. Existing imports from the three component packages remain
supported, while new examples can use one concise namespace.

The package also exports `run_tomography`, the end-to-end connection described
in `pipeline.md`. Core implementation remains in the three component packages;
this namespace contains no numerical backend logic and introduces no backend
dependency.

AI disclosure: this package interface and companion text were generated with
OpenAI Codex assistance on 2026-08-17. Independent review is pending.
