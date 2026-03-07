# Repository Guidelines

## Project Structure & Module Organization
This repository uses a `src/` layout. Shared models, import helpers, and evaluator plumbing live in `src/westworld/base.py`. The demo CLI is in `src/westworld/demo.py` and is exposed as `westworld-demo`. Site-specific task and evaluator modules are grouped by domain under `src/westworld/azora/`, `goodbuy/`, `megamart/`, `noodle_flights/`, and `travelpedia/`. Add new task logic inside the matching domain package and keep `src/westworld/__init__.py` limited to stable public exports.

## Build, Test, and Development Commands
Use Python 3.10+.

- `uv pip install -e ".[dev,datasets,playwright]"` installs the package with test, dataset, and browser extras.
- `python -m playwright install chromium` installs the browser required for demo and verifier flows.
- `westworld-demo --index 0` runs the first validation task locally.
- `python -m westworld.demo --task-id westworld/azora/basic_checkout/22` runs a specific task by ID.
- `python -m pytest` runs the test suite. No `tests/` tree is committed yet, so add tests with your change set.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for modules and functions, `PascalCase` for dataclasses and Pydantic models, and `UPPER_SNAKE_CASE` for module constants such as `HF_DATASET`. Keep explicit type hints on public functions and preserve concise docstrings where behavior is not obvious. Match the current import order: standard library, third-party, then local imports. No formatter or linter config is checked in, so keep edits consistent with surrounding files and avoid unrelated refactors.

## Testing Guidelines
Create tests under a top-level `tests/` package using `test_<module>.py` naming. Mirror the source layout when practical, for example `tests/test_base.py` or `tests/azora/test_basic_checkout.py`. Use `pytest` and `pytest-asyncio` for async evaluator coverage. Focus on task config generation, evaluator `update()` and `compute()` behavior, and failure paths; mock Playwright or remote dependencies unless end-to-end browser behavior is the target.

## Commit & Pull Request Guidelines
Recent commits use short prefixes such as `docs:` and `chore:` followed by an imperative summary, for example `docs: add evaluator usage example to README`. Continue that pattern for `feat:`, `fix:`, `docs:`, and `chore:` commits. Pull requests should explain the behavior change, list verification commands, link the relevant issue, and include screenshots or terminal output when demo flows or browser-visible behavior change.

## Security & Configuration Tips
Set `HALLUMINATE_API_KEY` in your shell environment instead of hardcoding secrets. If a change depends on dataset schema or external site behavior, document that assumption in the PR so evaluator regressions are easier to trace.
