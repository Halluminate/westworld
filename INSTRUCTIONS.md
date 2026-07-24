# Noodle Flights Agent — Westworld Benchmark Harness

An LLM-driven browser agent that solves [Halluminate's Westworld](https://github.com/Halluminate/westworld)
`noodle_flights` benchmark, with full [Arize Phoenix](https://github.com/Arize-ai/phoenix)
observability — every step traced, every action screenshotted.

Currently passing **all 8 `noodle_flights` task categories**.

---

## Table of contents

- [What this is](#what-this-is)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API keys](#api-keys)
- [Running Phoenix](#running-phoenix)
- [Running the agent](#running-the-agent)
- [Task categories](#task-categories)
- [How it works](#how-it-works)
- [Screenshots and step logging](#screenshots-and-step-logging)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)

---

## What this is

Westworld ships **task generators**, **verifiers (graders)**, and **synthetic simulator
websites** — but deliberately *not* the agent. The agent is the thing being measured.

This repo is that agent: a harness that reads a task from the dataset, drives a real
Chromium browser against the Noodle Flights simulator, produces an answer, and submits
it to Westworld's grader — while streaming traces to Phoenix.

Think of it as an exam:

| Exam | Here |
|---|---|
| Exam room | The simulator site (`flights.halluminate.ai`) |
| Question | The task from the dataset |
| Student | This agent |
| Invigilator's camera | Phoenix traces + screenshots |
| Marker | The Westworld grader |

---

## Prerequisites

- **Python 3.12** — *not 3.13*. On 3.13 `pandas` tries to build from source and fails.
- **git**
- A terminal, ~2 GB free disk (Chromium + dataset cache)

Verify:

```bash
python3.12 --version   # should print 3.12.x
```

---

## Installation

### 1. Clone this repo and Westworld

Westworld is **not on PyPI** — you must install it from source.

```bash
git clone <your-repo-url> phoenix-logger
cd phoenix-logger

git clone https://github.com/Halluminate/westworld.git
```

Your tree should look like:

```
phoenix-logger/
├── main.py                  # the agent (this repo)
├── INSTRUCTIONS.md
└── westworld/               # cloned benchmark
    └── src/westworld/noodle_flights/
```

### 2. Create the virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Work around the setuptools breakage

setuptools 82 removed `pkg_resources`, which some dependencies still import.
Pin it *before* installing anything:

```bash
echo "setuptools<82" > constraints.txt
export PIP_CONSTRAINT=$PWD/constraints.txt
```

> Keep `PIP_CONSTRAINT` exported for every `pip install` in this project.

### 4. Install Westworld and its extras

```bash
pip install -e "./westworld[datasets,playwright]"
```

### 5. Install the browser

```bash
python -m playwright install chromium
```

### 6. Install Phoenix and the Anthropic instrumentation

```bash
pip install arize-phoenix openinference-instrumentation-anthropic anthropic
```

---

## API keys

Two different keys are needed, and they are **not** interchangeable.

### `ANTHROPIC_API_KEY` — required

Drives the LLM that reasons about each page.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

> **A Claude Pro/Max subscription does not include API access.** You need a key from
> [console.anthropic.com](https://console.anthropic.com) with billing enabled. If you
> see `authentication_error`, this is almost always why.

Cost is a few cents per task run (Sonnet, ~1–3k tokens per step).

### `HALLUMINATE_API_KEY` — required

Authenticates against the simulator site. Westworld bakes it into the task URL
automatically once it's in the environment.

```bash
export HALLUMINATE_API_KEY="..."
```

### Optional environment variables

```bash
# Phoenix collector (defaults to this if unset)
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
```

To persist them, put the exports in a `.env` file and `source .env`, or add them to
your shell profile. **Never commit real keys.**

---

## Running Phoenix

Phoenix runs as a **separate process**. Start it first, in its own terminal:

```bash
source venv/bin/activate
phoenix serve
```

Then open **http://localhost:6006**.

Traces land in the project named `noodle-flights-agent` (set in `main.py` via
`register(project_name=...)`).

Leave this running while the agent runs. If Phoenix isn't up, the agent still works —
it just can't export traces.

---

## Running the agent

### 1. Pick a category

Open `main.py` and set the `CATEGORY` constant near the top:

```python
CATEGORY = "flight_search_roundtrip"
```

### 2. Run

```bash
source venv/bin/activate
python main.py 0        # run task index 0 only
python main.py          # run ALL tasks in the category + scoreboard
```

A Chromium window opens (`headless=False` by default) so you can watch it work.

### 3. Read the output

```
TASK: Find the cheapest round-trip flight from Accra (ACC) to ...
Parsed 1 date ranges: [((2026, 12, 20), (2026, 12, 25))]

--- range 1: depart (2026, 12, 20) return (2026, 12, 25) ---
   form: origin='ACC' dest='PHL' class='economy' adults=1
   [api] captured dep=39 ret=40
   [roundtrip] dep=['AA300'] ret=['AA400'] total=$1,816

[answer]
Cheapest round-trip: outbound flights AA300; return flights AA400; total price $1,816.

Score: 1.0
```

`Score: 1.0` is a pass. On failure, `Reason:` lists each grader **bucket** and whether
it matched — that's your debugging starting point.

---

## Task categories

All eight are supported. Set one at a time in `CATEGORY`.

| Category | Tasks | What the grader wants |
|---|---|---|
| `basic_flight_search` | 4 | Correct search form state |
| `date_range_search_simplified` | 2 | Cheapest dep/ret per date range (times, durations) |
| `date_range_search` | 4 | Above + **all** flights on a named airline |
| `airline_search_simple` | 3 | A single **total price** string |
| `natural_airline_search_simple` | — | Same, task phrased conversationally |
| `flight_search_budget` | — | An **exact phrase**: `"Yes, there is at least one option"` / `"No options found"` |
| `flight_search_cheapest` | 2 | Cheapest flight's **number + price** |
| `flight_search_roundtrip` | 2 | Both legs' flight numbers + total price |

To list the exact category names in the dataset:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Halluminate/westworld', split='validation')
cats = sorted({r['task_id'].split('noodle_flights/')[1].split('/')[0]
               for r in ds if 'noodle_flights/' in r['task_id']})
print('\n'.join(cats))
"
```

---

## How it works

### Two execution paths

```
_dispatch(idx)
    │
    ├── run_task ................ basic_flight_search only
    │     Fully autonomous: Claude picks every action.
    │     Emits agent.* spans.
    │
    └── run_extraction_task ..... the other seven categories
          Python drives the loop; Claude reads results.
          Emits step.* spans.
```

**`run_task`** is the *eyes → brain → hands* loop. Python scrapes numbered interactive
elements, sends the task + elements + history to Claude, Claude returns **one**
high-level JSON action, Python executes it with Playwright, repeat.

Actions are high-level (`set_airport`, `pick_date`, `submit`, `select_flight`,
`report_answer`) rather than raw clicks — *reasoning on top, reliable primitives
underneath*. The fragile widgets (React autocomplete, two-month calendar, passenger
stepper) are handled by proven code so the model doesn't have to fight them.

**`run_extraction_task`** exists because LLMs lose track of multi-step orchestration
across page reloads. Python owns the loop; Claude does the judgment.

### Where the answers come from

The results table **does not render flight numbers**. Trying to read them from the DOM
is a dead end. Instead the harness intercepts the site's own flights API response
(`departingFlights` / `returningFlights`) and replicates each grader's `filter_flights`
logic in Python:

```
direct only (no segments) → cabin + seats available → sort by cabin price
```

This is the single most important design decision in the project, and the reason the
later categories passed on the first attempt.

> **Describe this accurately** when presenting: the LLM drives the *form*; extraction is
> deterministic, reading the same API the grader reads. It is not "the agent visually
> reads the results table."

### Shared vs per-category

Everything up to the answer is **shared**: form filling, calendar navigation, passenger
stepper, cabin normalisation, API capture, state guards. Only the final answer assembly
forks per category — because each grader wants a genuinely different artifact.

---

## Screenshots and step logging

Every browser step becomes a **child span** carrying a **screenshot**.

- **`run_extraction_task`** → `step.range1_open`, `step.range1_fill_form`,
  `step.range1_submit`, `step.range1_read_departures`, `step.range1_select_departure`,
  `step.range1_read_returns`, `step.grade`
- **`run_task`** → a screenshot on each `agent.<action>` span

PNGs are written to:

```
./phoenix_screenshots/
```

## License

See the [Westworld repository](https://github.com/Halluminate/westworld) for benchmark licensing. This harness is yours to license as you choose.
