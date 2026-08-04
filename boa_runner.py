"""
boa_runner.py — Subprocess wrapper around BOA Constrictor's main.py

Handles:
  - Writing experiment YAML configs to disk
  - Running BOA train / compress / decompress via subprocess
  - Parsing stdout/stderr into a structured metrics dict

This module has no knowledge of sweep logic or plotting — it just
runs one BOA experiment and returns what it measured.
"""

import subprocess
import re
import os
import json
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def load_base_config(experiment_name: str) -> dict:
    """Load an existing experiment YAML as a config template.

    Args:
        experiment_name: Name of an already-completed BOA experiment.
                         Expects the file at experiments/<name>/<name>.yaml

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: if the YAML doesn't exist.
    """
    path = f"experiments/{experiment_name}/{experiment_name}.yaml"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Base experiment '{experiment_name}' not found at {path}.\n"
            f"Run a full experiment first so the YAML exists."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def write_config(config: dict, experiment_name: str) -> str:
    """Write a config dict to the experiments directory.

    Args:
        config:          Dict to serialise as YAML.
        experiment_name: Name for the new experiment directory.

    Returns:
        Path to the written YAML file.
    """
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    path = f"{exp_dir}/{experiment_name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path


def set_nested(d: dict, keys: list, value) -> None:
    """Set a value in a nested dict by a list of keys (in-place).

    Example:
        set_nested(cfg, ["model", "num_layers"], 4)
        # equivalent to cfg["model"]["num_layers"] = 4
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_nested(d: dict, keys: list):
    """Get a value from a nested dict by a list of keys.

    Raises KeyError if any key is missing.
    """
    for key in keys:
        d = d[key]
    return d


# ── Metric parsing ────────────────────────────────────────────────────────────

def parse_metrics(stdout: str) -> dict:
    """Parse BOA stdout/stderr into a structured metrics dict.

    All values default to None if the expected line isn't found —
    callers should handle None gracefully.

    Args:
        stdout: Combined stdout + stderr string from one or more BOA runs.

    Returns:
        Dict with keys:
            ratio_excl_model    float  compression ratio excluding model size
            ratio_incl_model    float  compression ratio including model size
            test_bpp            float  bits per byte on held-out test set
            final_val_bpp       float  val bpp from the last training epoch
            train_time_s        float  training wall-clock time in seconds
            compress_time_s     float  compression wall-clock time in seconds
            compress_mbps       float  compression throughput in MB/s
            decompress_time_s   float  decompression wall-clock time in seconds
            decompress_mbps     float  decompression throughput in MB/s
            compressed_bytes    int    size of compressed output in bytes
            model_params        int    number of model parameters
    """
    metrics: dict = {}

    def _float(pattern):
        m = re.search(pattern, stdout)
        return float(m.group(1)) if m else None

    def _int(pattern):
        m = re.search(pattern, stdout)
        return int(m.group(1).replace(",", "")) if m else None

    metrics["ratio_excl_model"]  = _float(r"Compression ratio \(excl\. model\): ([\d.]+)")
    metrics["ratio_incl_model"]  = _float(r"Compression ratio \(incl\. model\): ([\d.]+)")
    metrics["test_bpp"]          = _float(r"\[TEST\] bpp=([\d.]+)")
    metrics["train_time_s"]      = _float(r"Training complete in ([\d.]+)s")
    metrics["compress_time_s"]   = _float(r"Compression complete in ([\d.]+)s")
    metrics["compress_mbps"]     = _float(r"Compression complete in [\d.]+s \(([\d.]+) MB/s\)")
    metrics["decompress_time_s"] = _float(r"Decompression complete in ([\d.]+)s")
    metrics["decompress_mbps"]   = _float(r"Decompression complete in [\d.]+s \(([\d.]+) MB/s\)")
    metrics["compressed_bytes"]  = _int(r"Compressed size: ([\d,]+) bytes")
    metrics["model_params"]      = _int(r"Model parameters: ([\d,]+)")

    # val bpp: grab the last occurrence (final epoch)
    matches = re.findall(r"val bpp=([\d.]+)", stdout)
    metrics["final_val_bpp"] = float(matches[-1]) if matches else None

    return metrics


# ── BOA runner ────────────────────────────────────────────────────────────────

def run_boa(
    experiment_name: str,
    compress_only: bool = False,
    timeout: int = 3600,
    debug: bool = False,
) -> dict:
    """Run a BOA experiment and return parsed metrics.

    Args:
        experiment_name: Name of the experiment (YAML must already exist at
                         experiments/<name>/<name>.yaml).
        compress_only:   If True, skip training and run compress + decompress
                         as two separate subprocess calls. Use this when
                         sweeping params that don't affect the model
                         (e.g. chunks_count).
        timeout:         Per-subprocess timeout in seconds (default 1 hour).
        debug:           If True, print raw stdout lines containing key words.

    Returns:
        Metrics dict from parse_metrics().
    """
    label = " (compress/decompress only)" if compress_only else ""
    print(f"\n  Running BOA: {experiment_name}{label}")

    if compress_only:
        # BOA's --compress-only and --decompress-only are mutually exclusive,
        # so we run them as two separate calls and concatenate the output.
        r1 = subprocess.run(
            ["python3", "main.py", "--config", experiment_name, "--compress-only"],
            capture_output=True, text=True, timeout=timeout,
        )
        r2 = subprocess.run(
            ["python3", "main.py", "--config", experiment_name, "--decompress-only"],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = r1.stdout + r1.stderr + r2.stdout + r2.stderr

        if r1.returncode != 0 or r2.returncode != 0:
            print(f"  [WARN] BOA exit codes: compress={r1.returncode} decompress={r2.returncode}")
            for line in stdout.splitlines()[-10:]:
                print(f"    {line}")
    else:
        result = subprocess.run(
            ["python3", "main.py", "--config", experiment_name],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout + result.stderr

        if result.returncode != 0:
            print(f"  [WARN] BOA exit code: {result.returncode}")
            for line in stdout.splitlines()[-10:]:
                print(f"    {line}")

    if debug:
        keywords = ("ratio", "bpp", "complete", "compress", "decompress", "error", "warn")
        print("  [DEBUG] relevant output lines:")
        for line in stdout.splitlines():
            if any(k in line.lower() for k in keywords):
                print(f"    {line}")

    metrics = parse_metrics(stdout)
    print(
        f"  → ratio_excl={metrics.get('ratio_excl_model')}, "
        f"compress={metrics.get('compress_mbps')} MB/s, "
        f"decompress={metrics.get('decompress_mbps')} MB/s"
    )
    return metrics


#if json file is provided then we use it!
#everything now simplifies since we can simply just load all the json entries and pass it into boa_sweep

def load_runs_json(path: str) -> list:
    """Load a runs.json file as produced by the new BOA version."""
    with open(path) as f:
        return json.load(f)


def extract_metrics_from_run(entry: dict) -> dict:
    """Extract our standard metrics dict from one runs.json entry.

    Maps the new JSON schema to the same keys the rest of the pipeline
    expects, so boa_sweep.py and boa_plot.py need no changes.

    Args:
        entry: One top-level entry from runs.json (with 'name', 'training', 'runs').

    Returns:
        Metrics dict with the same keys as parse_metrics() used to return.
    """
    training = entry.get("training") or {}
    runs = entry.get("runs") or []

    # Pick the most recent GPU run for throughput numbers
    gpu_runs = [r for r in runs if r.get("engine") in ("gpu", "int-gpu")]
    latest = max(gpu_runs, key=lambda r: r.get("ts", ""), default={})

    return {
        "ratio_excl_model":  training.get("ratio"),
        "ratio_incl_model":  None,          # not in new schema
        "test_bpp":          training.get("test_bpb"),
        "final_val_bpp":     None,          # not in new schema
        "train_time_s":      training.get("wall_min", 0) * 60 if training.get("wall_min") else None,
        "compress_time_s":   None,          # not directly available
        "compress_mbps":     latest.get("gpu_enc"),
        "decompress_time_s": None,
        "decompress_mbps":   latest.get("gpu_dec"),
        "compressed_bytes":  None,
        "model_params":      training.get("params"),
        # Extra fields from the new schema worth keeping
        "backbone":          training.get("backbone"),
        "d_model":           training.get("d_model"),
        "num_layers":        training.get("num_layers"),
        "lossless":          latest.get("lossless"),
        "streams":           latest.get("streams"),
        "dataset_mib":       training.get("dataset_mib"),
    }
