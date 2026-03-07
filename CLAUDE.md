# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Halluminate Westworld is a benchmark for evaluating web agents on realistic websites. It provides task definitions and evaluators that verify whether an agent correctly completed a web task (e.g., filling a checkout form, searching for flights). The dataset is hosted on HuggingFace (`Halluminate/westworld`).

## Commands

```bash
# Install (editable, with optional deps)
uv pip install -e ".[datasets,playwright]"
python -m playwright install chromium

# Run demo (human-in-the-loop)
westworld-demo --task-id westworld/azora/basic_checkout/22
westworld-demo --index 0

# Run tests
uv run pytest

# Environment variable required for simulated environments
export HALLUMINATE_API_KEY=your-key-here
```

## Architecture

### Core (`src/westworld/base.py`)

- **`DatasetItem`** — Pydantic model representing a row from the HuggingFace dataset. Contains `task_id`, `task_generation_config_json` (JSON-encoded instantiation config), and metadata fields (`env`, `domain`, `l1_category`, `l2_category`, difficulty, etc.).
- **`BaseTaskConfig`** — Pydantic model with `task` (instruction text), `url`, `user_metadata`, and `eval_config` (dict with `_target_` for evaluator instantiation).
- **`BaseMetric`** — Abstract evaluator interface with `update(**kwargs)`, `compute()`, and `reset()` methods. All evaluators subclass this.
- **`instantiate(config)`** — Hydra-like recursive instantiation. Dicts with `_target_` key are resolved via `omni_import` and constructed with remaining keys as kwargs. Supports `_args_`, `_recursive_: False`.
- **`omni_import(path)`** — Dynamic import by dotted path string (e.g., `westworld.azora.basic_checkout.HalluminateAzoraBasicCheckout`).

### Evaluation flow

1. Load a `DatasetItem` from the HuggingFace dataset
2. Call `dataset_item.generate_task_config()` — this calls `instantiate()` on the JSON config, which invokes the domain-specific `generate_task_config()` function
3. The returned `BaseTaskConfig.eval_config` contains a `_target_` pointing to the evaluator class
4. Call `instantiate(task_config.eval_config)` to get an evaluator instance
5. During the agent loop: `await evaluator.update(page=page)` on each step
6. After completion: `await evaluator.compute()` returns a result with `score`, `reasoning`, etc.

### Domain modules (`src/westworld/<domain>/`)

Each domain (website) has its own package with task-specific modules. Each module contains:

- **Domain dataclasses** — model the expected state (e.g., `ContactInfo`, `FlightSearchParams`, `ApplicationState`)
- **Verification functions** — compare expected vs actual state extracted from the page
- **Evaluator class** (subclass of `BaseMetric`) — uses Playwright `page.evaluate()` to extract state via `window.__testingAPI`, then runs verification. Returns a `FinalResult` with score 0.0 or 1.0
- **`generate_task_config()`** function — renders date formulas, replaces `{HALLUMINATE_API_KEY}` in URLs, builds the `BaseTaskConfig`

Current domains:
- **azora** — e-commerce (checkout, delivery instructions)
- **goodbuy** — e-commerce (checkout, delivery, pickup)
- **megamart** — e-commerce (checkout, delivery, pickup)
- **noodle_flights** — travel/flights (various search types)
- **travelpedia** — travel (flights, hotels)

### Key patterns

- Evaluators extract page state via `page.evaluate("() => window.__testingAPI.getState()")` or similar JS calls to the simulated site's testing API
- Date fields in task configs use formula strings like `"now() + timedelta(days=120)"` that are evaluated at task generation time relative to user timezone/timestamp
- Task instruction text uses `{departureDate}` / `{returnDate}` placeholders rendered to human-readable format
- URLs contain `{HALLUMINATE_API_KEY}` placeholder replaced from environment variable
- The `_target_` instantiation pattern is used throughout — similar to Hydra but custom

## Package structure

- Build system: hatchling
- Source layout: `src/westworld/`
- Python: >=3.10
- Core deps: pydantic, loguru, aiofiles
- Optional deps: `datasets` (HuggingFace), `playwright` (browser automation)
- Entry point: `westworld-demo` -> `westworld.demo:main`
- No tests exist in the repo yet
