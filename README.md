# Halluminate Westworld

A benchmark for web agents to perform tasks on realistic websites.

Dataset: https://huggingface.co/datasets/Halluminate/westworld

Blog post: https://halluminate.ai/blog/westworld

## Quick Start: Try a Task Yourself

Want to understand what the benchmark tasks look like? You can run them manually using our human-in-the-loop demo:

### Step 1: Install with Browser Support

```bash
# Using uv (recommended)
uv pip install -e ".[datasets,playwright]"
python -m playwright install chromium

# Or using pip
pip install -e ".[datasets,playwright]"
python -m playwright install chromium
```

### Step 2: Set Your API Key (for simulated environments)

```bash
export HALLUMINATE_API_KEY=your-key-here
```
*contact wyatt@halluminate.ai for an api key*

### Step 3: Run the Demo

You can run the demo in two ways:

**Option A: Run by dataset index**
```bash
westworld-demo --index 0
```

**Option B: Run by specific task ID**
```bash
westworld-demo --task-id westworld/azora/basic_checkout/22
```

**Alternative: Run as Python module**
```bash
# By index
python -m westworld.demo --index 0

# By task ID
python -m westworld.demo --task-id westworld/azora/basic_checkout/22
```

## Usage

### Dataset

The benchmark dataset is available on [HuggingFace](https://huggingface.co/datasets/Halluminate/westworld):

```python
from datasets import load_dataset

dataset = load_dataset("Halluminate/westworld")
```

### Loading and Evaluating Tasks

```python
from westworld.base import DatasetItem, instantiate

# Load a task from the dataset
task_item = DatasetItem(**dataset[0])

# Generate the task configuration
task_config = task_item.generate_task_config()

# Access task details
print(f"Task: {task_config.task}")
print(f"URL: {task_config.url}")
print(f"Evaluation Config: {task_config.eval_config}")

# Instantiate evaluator when starting the agent task
agent = ...
evaluator = instantiate(task_config.eval_config)

for _ in range(max_steps):
    # Agent takes a step
    ...

    # Update evaluator
    await evaluator.update(...)

# Get the final evaluation result
eval_result = await evaluator.compute()
```
*Note: most evaluators rely on site state for verification, so ensure the verifier is run before closing the browser window*

### Task Categories

The benchmark includes the following task categories (L1 categories):

- **e_commerce**: Online shopping tasks across multiple platforms
  - Basic checkout flows
  - Delivery instruction handling
  - Pickup order management

- **travel**: Travel booking and search tasks
  - Flight searches (basic, roundtrip, date ranges)
  - Airline-specific searches
  - Hotel searches
  - Budget-constrained searches


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Halluminate Westworld in your research, please cite:

```bibtex
@software{halluminate_westworld,
  title = {Halluminate Westworld: A Web Agent Benchmark},
  author = {Halluminate},
  year = {2025},
  url = {https://github.com/Halluminate/westworld}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/Halluminate/westworld/issues).
