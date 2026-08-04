"""
boa_sweep.py — Parameter sweep orchestration for BOA Constrictor

Sweeps a single BOA YAML parameter across a range of values,
running train+compress+decompress (or compress+decompress only)
for each value and collecting metrics into a JSON results file.

This module is intentionally decoupled from plotting — it only
produces the JSON. Pass --plot-only to plot without re-running.

Usage:
    python3 boa_sweep.py --param chunks_count --values 100 250 500 --compress-only
    python3 boa_sweep.py --param num_layers --values 1 2 4 6
    python3 boa_sweep.py --param num_layers --values 1 2 4 6 --plot-only
"""

import copy
import json
import os
import subprocess
import argparse
from datetime import datetime


from boa_plot import plot_sweep
from boa_runner import load_runs_json, extract_metrics_from_run, load_base_config, write_config, set_nested, run_boa



# ── Configuration — edit these to match your setup ───────────────────────────

# Base experiment to use as the config template.
# Must already exist at experiments/<name>/<name>.yaml
BASE_EXPERIMENT = "test_bin_data"

# Dataset for all sweep runs — must be a flat binary file (not ROOT, not HDF5)
DATASET_PATH = "/eos/home-j/jsurduto/boa/rntuple_flat.bin"


# Where results are saved between runs — survives session dropout
RESULTS_FILE = f"/eos/home-j/jsurduto/boa/sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# list of fields we can group/filter by:
SWEEP_FIELDS = {
    "chunks_count",   # kept for backward compat with old log-scrape runs
    "num_layers",
    "d_model",
    "backbone",
    "streams",
    "epochs",
}

PARAM_MAP = {
    "num_layers":   ["model",       "num_layers"],
    "d_model":      ["model",       "d_model"],
    "epochs":       ["training",    "epochs"],
    "seq_len":      ["dataloader",  "seq_len"],
    "batch_size":   ["dataloader",  "batch_size"],
    "lr":           ["training",    "lr"],
    "chunks_count": ["compression", "chunks_count"],
    "streams":      ["compression", "streams"],
}

#fields we don't want to compare in json configs
IGNORE_FIELDS = { 
    "seed", "params", "ts", "git_commit", "ckpt", "config",
    "host", "gpu", "torch_version", "env", "wall_min",
    "train_mibs", "test_bpb", "ratio",
}

# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(results: list, path: str) -> None:
    """Write results list to JSON. Overwrites the file atomically."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {path}")


def load_results(path: str) -> list:
    """Load existing results JSON, or return empty list if file doesn't exist or is empty."""
    if os.path.exists(path):
        with open(path) as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    return []


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(
    param_name: str,
    values: list,
    skip_existing: bool = True,
    compress_only: bool = False,
) -> list:
    """Run a parameter sweep and return the full results list.

    For each value in `values`:
      1. Deep-copy the base experiment config.
      2. Override the target parameter.
      3. Write a new experiment YAML.
      4. Run BOA (full pipeline, or compress+decompress only).
      5. Append metrics to the results JSON immediately (crash-safe).

    Args:
        param_name:     Key in PARAM_MAP — the parameter to vary.
        values:         List of values to sweep over.
        skip_existing:  Skip values already present in the results JSON.
                        Set False (--no-resume) to re-run everything.
        compress_only:  If True, use a pre-trained model and only run
                        compress + decompress. Training is skipped entirely.
                        The model_path from the base config is preserved.
                        Use this for chunks_count sweeps where retraining
                        would be wasteful and wrong.

    Returns:
        Full results list (all params, all runs, including previously saved).
    """
    if param_name not in PARAM_MAP:
        raise ValueError(
            f"Unknown parameter '{param_name}'. "
            f"Valid options: {list(PARAM_MAP.keys())}"
        )

    yaml_keys = PARAM_MAP[param_name]

    print(f"\n{'='*60}")
    print(f"BOA Parameter Sweep: {param_name}")
    print(f"Values:              {values}")
    print(f"Base experiment:     {BASE_EXPERIMENT}")
    print(f"Dataset:             {DATASET_PATH}")
    print(f"Compress-only:       {compress_only}")
    print(f"{'='*60}")

    base_config = load_base_config(BASE_EXPERIMENT)
    base_config["file_path"] = DATASET_PATH

    all_results = load_results(RESULTS_FILE)
    already_done = {(r["param_name"], r["param_value"]) for r in all_results}

    for value in values:
        key = (param_name, value)

        if skip_existing and key in already_done:
            print(f"\n  Skipping {param_name}={value} (already in results)")
            continue

        print(f"\n{'─'*40}")
        print(f"  {param_name} = {value}")

        cfg = copy.deepcopy(base_config)
        set_nested(cfg, yaml_keys, value)

        if compress_only:
            # Keep model_path from base config so BOA loads the existing
            # trained model rather than retraining from scratch.
            pass
        else:
            # Remove model_path so BOA trains a fresh model for this config.
            cfg.pop("model_path", None)

        exp_name = f"sweep_{param_name}_{value}"
        cfg["name"] = exp_name
        write_config(cfg, exp_name)

        try:
            metrics = run_boa(exp_name, compress_only=compress_only)
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Timed out: {param_name}={value}")
            metrics = {}
        except Exception as e:
            print(f"  [ERROR] Failed: {param_name}={value}: {e}")
            metrics = {}

        result = {
            "param_name":  param_name,
            "param_value": value,
            "yaml_keys":   yaml_keys,
            "experiment":  exp_name,
            "metrics":     metrics,
        }
        all_results.append(result)
        # Save immediately so a mid-sweep dropout doesn't lose prior work
        save_results(all_results, RESULTS_FILE)

    print(f"\n{'='*60}")
    print(f"Sweep complete. {len(values)} values processed.")
    print(f"{'='*60}")

    return all_results

#there are now two other different modes. 
'''
1. either compare different experiments across different parameters
here we check that the non-ignored parameters are the same and then compare the outputs
2. as before, compare a sweep across one parameter over one experiment
here we compare that the non-ignored parameters are the same apart from what is being changed. and then compare the outputs.
in theory, these two modes could be combined into a joint experiment a change + experiment b change. but that can wait until v3.
'''
def compare_experiments(
    runs_json_path: str,
    compare_params: list,
    results_file: str,
) -> list:

    """Compare distinct experiments along one or more output metrics.

    Verifies that all entries share the same config (excluding IGNORE_FIELDS
    and the compare_params themselves), then extracts metrics for plotting.

    Args:
        runs_json_path:  Path to runs.json.
        compare_params:  List of output fields to compare across experiments
                         e.g. ["ratio_excl_model", "decompress_mbps"]
        results_file:    Where to save results JSON.

    Returns:
        Results list where param_name="experiment" and param_value=entry name.
    """

    entries = [e for e in load_runs_json(runs_json_path) if e.get("training")]

    if len(entries) < 2:
        raise ValueError(f"Need at least 2 entries with training blocks to compare, got {len(entries)}")


    training_blocks = [e["training"] for e in entries]
    all_keys = set().union(*training_blocks) - IGNORE_FIELDS - set(compare_params) 

    mismatches = {}  #compare keys but not the ignored ones (ones that we expect to be different)
    for key in all_keys:
        values = [t.get(key) for t in training_blocks]
        if len(set(str(v) for v in values)) > 1:
            mismatches[key] = values

    if mismatches:
        print("\n  [WARN] The following config fields differ between experiments:")
        for key, vals in mismatches.items():
            names = [e["name"] for e in entries]
            for name, val in zip(names, vals):
                print(f"    {name}: {key} = {val}")
        print("  Proceeding anyway — verify this is intentional.\n")
    else:
        print("  Config check passed — all non-ignored fields match across experiments.")

    all_results = load_results(results_file)
    already_done = {(r["param_name"], r["param_value"]) for r in all_results}

    for entry in entries:
        key = ("experiment", entry["name"])
        if key in already_done:
            print(f"  Skipping '{entry['name']}' (already in results)")
            continue

        metrics = extract_metrics_from_run(entry)
        print(f"  {entry['name']}: "
              + ", ".join(f"{p}={metrics.get(p)}" for p in compare_params))

        result = {
            "param_name":  "experiment",
            "param_value": entry["name"],
            "yaml_keys":   [],
            "experiment":  entry["name"],
            "metrics":     metrics,
            "compare_params": compare_params,
        }
        all_results.append(result)

    save_results(all_results, results_file)
    return all_results

#and here's the sweep mode from before -- just simplified because now we have Akshat's json
def run_sweep_from_json(
    runs_json_path: str,
    param_name: str,
    results_file: str,
) -> list:
    entries = load_runs_json(runs_json_path)
    all_results = load_results(results_file)
    already_done = {(r["param_name"], r["param_value"]) for r in all_results}

    for entry in entries:
        training = entry.get("training") or {}
        # Check training block first, then run blocks, then top level
        value = training.get(param_name)
        if value is None:
            # Check individual run entries (e.g. streams lives in runs[])
            runs = entry.get("runs") or []
            gpu_runs = [r for r in runs if r.get("engine") in ("gpu", "int-gpu")]
            latest = max(gpu_runs, key=lambda r: r.get("ts", ""), default={})
            value = latest.get(param_name)
        if value is None:
            print(f"  Skipping '{entry['name']}' — no field '{param_name}' found")
            continue

        key = (param_name, value)
        if key in already_done:
            print(f"  Skipping {param_name}={value} (already in results)")
            continue

        metrics = extract_metrics_from_run(entry)
        print(f"  {entry['name']}: {param_name}={value} "
              f"ratio={metrics.get('ratio_excl_model')} "
              f"gpu_dec={metrics.get('decompress_mbps')} MB/s")

        result = {
            "param_name":  param_name,
            "param_value": value,
            "yaml_keys":   [],
            "experiment":  entry["name"],
            "metrics":     metrics,
        }
        all_results.append(result)

    save_results(all_results, results_file)
    return all_results



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep a BOA YAML parameter and collect compression metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sweep chunks_count without retraining (compress/decompress only):
  python3 boa_sweep.py --param chunks_count --values 100 250 500 1000 --compress-only

  # Sweep model depth (retrains for each value):
  python3 boa_sweep.py --param num_layers --values 1 2 4 6

  # Just plot existing results without running anything:
  python3 boa_sweep.py --param chunks_count --values 100 250 500 1000 --plot-only
        """,
    )
    p.add_argument("--param", required=False, default=None, choices=list(PARAM_MAP.keys()),
                   help="Parameter to sweep")
    p.add_argument("--values", nargs="+", type=float, required=False, default=None,
                   help="Values to sweep over")
    p.add_argument("--compress-only", action="store_true",
                   help="Skip training; use existing model for compress+decompress only. "
                        "Correct for sweeping chunks_count.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip running BOA; just plot results from RESULTS_FILE")
    p.add_argument("--plot-data-path", default=None,
                    help="Path to export plot data JSON. Defaults to plot_data_<param>.json")                 
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all values even if already in results JSON")
    p.add_argument("--output-plot", default=None,
                   help="Path to save the plot PNG. Defaults to sweep_<param>.png in EOS")
    p.add_argument("--debug", action="store_true",
                   help="Print raw BOA output lines for debugging")
    p.add_argument("--runs-json", default=None,
                    help="Path to a runs.json file from BOA. If provided, reads metrics "
                    "directly instead of running BOA via subprocess.")
    p.add_argument("--compare-experiments",action="store_true",
                    help="Mode: compare distinct experiments against each other. "
                        "Verifies configs match and plots --param values side by side. "
                        "Requires --runs-json.")
    p.add_argument("--compare-params",nargs="+", default=["ratio_excl_model", "decompress_mbps"],
                    help="Output metrics to compare across experiments when using "
                        "--compare-experiments. "
                        "Defaults to ratio_excl_model and decompress_mbps. "
                        "Example: --compare-params ratio_excl_model decompress_mbps test_bpp")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    param = args.param
    if not args.compare_experiments and not args.param:
        print("--param is required unless using --compare-experiments")
        raise SystemExit(1)
    # Cast to int for discrete params, keep float for lr
    int_params = {"num_layers", "d_model", "epochs", "seq_len", "batch_size", "chunks_count", "streams"}
    values = [int(v) for v in args.values] if (args.values and param in int_params) else (args.values or [])
    
    if not args.plot_only:
        if args.compare_experiments:
            if not args.runs_json:
                print("--compare-experiments requires --runs-json")
                raise SystemExit(1)
            all_results = compare_experiments(
                runs_json_path=args.runs_json,
                compare_params=args.compare_params,
                results_file=RESULTS_FILE,
            )   
            param = "experiment" #the 'param' for experiment comparison
        elif args.runs_json:
            if not param:
                print("--runs-json requires --param (the field to sweep over)")
                raise SystemExit(1)
            all_results = run_sweep_from_json(
                runs_json_path=args.runs_json,
                param_name=param,
                results_file=RESULTS_FILE,
            )
        else:
            all_results = run_sweep(
                param_name=param,
                values=values,
                skip_existing=not args.no_resume,
                compress_only=args.compress_only,
            )
    else:
        all_results = load_results(RESULTS_FILE)
        if not all_results:
            print(f"No results found at {RESULTS_FILE}. Run without --plot-only first.")
            raise SystemExit(1)
    

    output_path = args.output_plot or f"/eos/home-j/jsurduto/boa/sweep_{param}.png"
    plot_data_path = args.plot_data_path or f"/eos/home-j/jsurduto/boa/plot_data_{param}.json"
    plot_sweep(all_results, param_name=param, output_path=output_path,
               plot_data_path=plot_data_path)
