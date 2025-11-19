#!/usr/bin/env python
"""
demo.py

Human-in-the-loop demo of Westworld benchmark where:

- The human + Playwright browser act as the "agent loop".
- Each page navigation is treated as an agent step.
- We call evaluator.update(...) on every step.
- At the end, we call evaluator.compute() for the final score.

This demonstrates how you'd integrate Westworld evaluators into
an agent loop, just with a human standing in for the policy.

Usage:
    # Install with playwright support
    pip install westworld[playwright]
    python -m playwright install chromium

    # Set your API key (for simulated environments)
    export HALLUMINATE_API_KEY=your-key-here

    # Run the demo
    python -m westworld.demo --task-id westworld/azora/basic_checkout/0

    # Or load by index from HuggingFace
    python -m westworld.demo --dataset Halluminate/westworld --index 0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Dict

from westworld.base import DatasetItem, instantiate

HF_DATASET = "Halluminate/westworld"
HF_SPLIT = "validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human-in-the-loop demo for Westworld benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", help="Task ID to load (e.g., westworld/azora/basic_checkout/0)")
    parser.add_argument("--dataset", default=HF_DATASET, help=f"HuggingFace dataset name (default: {HF_DATASET})")
    parser.add_argument("--split", default=HF_SPLIT, help=f"Dataset split (default: validation)")
    parser.add_argument("--index", type=int, help="Dataset index to load")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    return parser.parse_args()


def load_task(task_id: str, dataset: str = HF_DATASET, split: str = HF_SPLIT) -> Dict[str, Any]:
    """Load a task row by task_id from the dataset."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("Error: datasets package required. Install with: pip install westworld[datasets]")
        sys.exit(1)

    print(f"Loading dataset: {dataset} (split: {split})")
    ds = hf_load_dataset(dataset, split=split)

    for row in ds:
        if row["task_id"] == task_id:
            return row

    raise ValueError(f"Task {task_id} not found in {dataset}/{split}")


def load_task_by_index(index: int, dataset: str = HF_DATASET, split: str = HF_SPLIT) -> Dict[str, Any]:
    """Load a task row by index from the dataset."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("Error: datasets package required. Install with: pip install westworld[datasets]")
        sys.exit(1)

    print(f"Loading dataset: {dataset} (split: {split})")
    ds = hf_load_dataset(dataset, split=split)

    if index >= len(ds):
        raise ValueError(f"Index {index} out of range (dataset has {len(ds)} items)")

    return ds[index]


async def attach_human_agent_loop(page, evaluator) -> None:
    """
    This is the human-agent-loop. Execute the task by navigating the website.
    Each navigation triggers evaluator.update().
    """

    async def on_navigation():
        try:
            await evaluator.update(page=page)
        except Exception as e:
            print(f"[WARN] evaluator.update(page={page.url!r}) failed: {e}")

    page.on("framenavigated", lambda frame: asyncio.create_task(on_navigation()))


async def run_human_session(task_data: Dict[str, Any], headless: bool = False) -> None:
    """Run a human-in-the-loop demo session."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: playwright not installed. Install with:")
        print("  pip install westworld[playwright]")
        print("  python -m playwright install chromium")
        sys.exit(1)

    # Generate the task config from the dataset item
    dataset_item = DatasetItem.model_validate(task_data)
    task_config = dataset_item.generate_task_config()

    # Instantiate the Westworld evaluator
    evaluator = instantiate(task_config.eval_config)

    print("\n" + "=" * 80)
    print("TASK")
    print("=" * 80)
    print(f"Task ID: {task_data.get('task_id', 'Unknown')}")
    print(f"URL:     {task_config.url}")
    print(f"Task:    {task_config.task}")
    print("=" * 80 + "\n")

    input("Press Enter when ready to start the browser...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1600, "height": 900})
        page = await context.new_page()

        await page.goto(task_config.url, timeout=60_000, wait_until="load")

        print(
            "\nBrowser opened.\n"
            "➡ You are now the agent.\n"
            "➡ Follow the instructions above to complete the task.\n"
            "➡ When done, press ENTER in this terminal (do not close the browser).\n"
        )

        # Reset the evaluator and update with the page object
        await evaluator.reset()
        await evaluator.update(page=page)
        await attach_human_agent_loop(page, evaluator)

        # Wait for user to press Enter when task is complete
        await asyncio.to_thread(input, "\nPress Enter when you've completed the task... ")

        # Final update before computing result
        try:
            await evaluator.update(page=page)
        except Exception as e:
            print(f"[WARN] Final evaluator.update(page=...) failed: {e}")

        # Compute the evaluation result while page is still open
        print("\nComputing evaluation result...\n")
        result = await evaluator.compute()

        # Now we can close the browser
        await context.close()
        await browser.close()

    # Report the result

    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Score: {getattr(result, 'score', None)}")
    if hasattr(result, "reasoning"):
        print(f"Reasoning: {result.reasoning}")
    if hasattr(result, "details"):
        print(f"Details: {result.details}")
    print("=" * 80 + "\n")


async def async_main():
    """Async main function."""
    args = parse_args()

    # Determine how to load the task
    if args.task_id:
        task_data = load_task(args.task_id, dataset=args.dataset, split=args.split)
    elif args.index is not None:
        task_data = load_task_by_index(args.index, dataset=args.dataset, split=args.split)
    else:
        print("Error: Must provide either --task-id or --index")
        sys.exit(1)

    await run_human_session(task_data, headless=args.headless)


def main():
    """Entry point for the westworld-demo command."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
