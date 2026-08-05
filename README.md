# BOA_plotting_pipeline

Parameter sweep and plotting tooling for [BOA Constrictor](https://github.com/boa-collaboration/boa-constrictor).
Credit to [https://github.com/ajbyrnes/LossBench/](https://github.com/ajbyrnes/LossBench/tree/main) for the examples and motivation as well as Claude for much of the formatting and test files. 
## Files

| File | Purpose |
|------|---------|
| `boa_runner.py` | Subprocess wrapper around BOA's `main.py`. Runs experiments and parses stdout into metrics. |
| `boa_sweep.py` | Sweep orchestration. Loops over parameter values, calls `boa_runner`, saves results to JSON. |
| `boa_plot.py` | Assembles plot panels into a figure from a results list. |
| `plot_library.py` | Individual plot function definitions. Add a new function here to add a new panel. |
| `test_boa_sweep.py` | Unit tests for all of the above. |
| `conftest.py` | Empty — tells pytest to add this directory to `sys.path`. |

## Configuration

Edit the constants at the top of `boa_sweep.py`:

```python
BASE_EXPERIMENT = "your_experiment_name"   # template YAML — must already exist
DATASET_PATH    = "/path/to/data.bin"
RESULTS_FILE    = "/path/to/results.json"
```

## Usage

Run from inside the `boa-constrictor` directory with the venv active.

```bash
# Sweep chunks_count without retraining (compress/decompress only):
python3 BOA_plotting_pipeline/boa_sweep.py --param chunks_count --values 100 250 500 1000 --compress-only

# Sweep model depth (retrains for each value):
python3 BOA_plotting_pipeline/boa_sweep.py --param num_layers --values 1 2 4 6

# Replot existing results without re-running anything:
python3 BOA_plotting_pipeline/boa_sweep.py --param chunks_count --values 100 250 500 --plot-only

# Re-run everything from scratch ignoring saved results:
python3 BOA_plotting_pipeline/boa_sweep.py --param num_layers --values 1 2 4 6 --no-resume
```

## Adding a new plot

1. Add a function to `plot_library.py` with signature `def my_plot(ax, sweep_results, param_name)`
2. Add it to the relevant `PLOTS_*` list (or `PLOTS_DEFAULT`) at the bottom of `plot_library.py`
3. Done — the grid resizes automatically

## Tests

```bash
python3 -m pytest BOA_plotting_pipeline/test_boa_sweep.py -v
```

No GPU or BOA installation required — all subprocess calls are mocked.
