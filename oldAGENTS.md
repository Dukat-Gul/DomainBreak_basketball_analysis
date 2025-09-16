# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: entry point for end‑to‑end analysis.
- Core modules: `drawers/`, `trackers/`, `team_assigner/`, `speed_and_distance_calculator/`, `shot_detector/`, `shot_classifier/`, `shot_visualizer/`, `pass_and_interception_detector/`, `ball_acquisition/`, `tactical_view/`, `tactical_view_converter/`, `utils/`.
- Data & assets: `input_videos/`, `output_videos/`, `images/`, `models/`, `configs/`, `roboflow_dataset/`.
- Experiments & docs: `training_notebooks/`, `docs/`, `stubs/`.
- Tests: root‐level `test_*.py` files plus reference images `test_*_output.jpg`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`).
- Install deps: `pip install -r requirements.txt`.
- Run pipeline: `python main.py` (use configs in `configs/` as needed).
- Evaluate/inspect: `python evaluation.py`, `python debug_annotations.py`.
- Tests (pytest-style): `pytest -q` or a subset `pytest -k drawer`.

## Coding Style & Naming Conventions
- Python 3; use 4‑space indentation; line length ≤ 100.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Imports: standard lib → third‑party → local (grouped with blank lines).
- Type hints for new/changed functions; docstrings for public APIs.
- Prefer pure, testable functions in `utils/`; keep script‑only logic in `main.py`.

## Testing Guidelines
- Framework: pytest; place tests as `test_*.py` at repo root or next to modules.
- Add tests for new logic and edge cases (e.g., empty frames, variable FPS).
- Update or regenerate reference images when rendering changes; include reasoning in PR.
- Quick run examples: `pytest test_court_drawer.py -q`, `pytest -k tracker -q`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, short scope first line (e.g., `drawers: fix arc rendering`).
- Reference issues: `Fixes #123` or `Refs #123`.
- PRs must include: concise description, motivation, before/after screenshots or short video for visual changes, test coverage notes, and reproduction steps.
- Keep PRs focused and small; note any non‑functional refactors.

## Security & Configuration Tips
- Store API keys (e.g., Roboflow) in environment variables; do not commit secrets.
- Large assets and outputs should go to `output_videos/` and be git‑ignored if needed.
