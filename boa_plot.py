"""
boa_plot.py — Assembles plot panels from plot_library into a figure.

Adding a new plot: add a function to plot_library.py and register it
in PLOT_REGISTRY. Nothing here needs to change.
"""

import math
import os
import json
import matplotlib.pyplot as plt
from typing import Optional

from plot_library import PLOT_REGISTRY, PLOTS_DEFAULT, _extract

# Dataset label used in the plot title — set to match boa_sweep.py

DATASET_LABEL = 'rntuple_flat.bin'

def _safe(values: list, default: float = 0.0) -> list:
    """Replace None entries with a default so matplotlib doesn't crash."""
    return [v if v is not None else default for v in values]

def plot_sweep(
    results: list,
    param_name: str,
    output_path: Optional[str] = None,
    dataset_label: str = DATASET_LABEL,
    plot_data_path: Optional[str] = None,
) -> None:
    """Assemble and render the plot set for a parameter sweep.

    Looks up the plot set for param_name in PLOT_REGISTRY (falls back to
    PLOTS_DEFAULT). Lays them out in a grid automatically — 2 columns,
    as many rows as needed.
    """
    # Filter and validate
    sweep_results = [
        r for r in results
        if r["param_name"] == param_name
        and r["metrics"].get("ratio_excl_model") is not None
    ]
    if not sweep_results:
        print(f"No complete results for '{param_name}'.")
        return

    sweep_results.sort(key=lambda r: r["param_value"])

    # Export plot data if requested
    if plot_data_path:
        save_plot_data(sweep_results, param_name, plot_data_path)

    # Get plot functions for this parameter
    plot_fns = PLOT_REGISTRY.get(param_name, PLOTS_DEFAULT)
    n = len(plot_fns)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    # Flatten axes so we can index them linearly regardless of shape
    axes_flat = axes.flatten() if n > 1 else [axes]

    fig.suptitle(
        f"BOA Constrictor — sweep over '{param_name}'\nDataset: {dataset_label}",
        fontsize=14,
    )

    for i, fn in enumerate(plot_fns):
        fn(axes_flat[i], sweep_results, param_name)

    # Hide any unused axes (if n is odd)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def save_plot_data(sweep_results, param_name, output_path):
    """Export filtered sorted plot data to JSON, appending to existing file."""
    export = {
        "param_name": param_name,
        "points": [
            {
                "param_value":       r["param_value"],
                "experiment":        r["experiment"],
                "ratio_excl_model":  r["metrics"].get("ratio_excl_model"),
                "ratio_incl_model":  r["metrics"].get("ratio_incl_model"),
                "compress_mbps":     r["metrics"].get("compress_mbps"),
                "decompress_mbps":   r["metrics"].get("decompress_mbps"),
                "compress_time_s":   r["metrics"].get("compress_time_s"),
                "decompress_time_s": r["metrics"].get("decompress_time_s"),
                "test_bpp":          r["metrics"].get("test_bpp"),
                "train_time_s":      r["metrics"].get("train_time_s"),
                "model_params":      r["metrics"].get("model_params"),
            }
            for r in sweep_results
        ],
    }
    existing = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    for i, entry in enumerate(existing):
        if entry.get("param_name") == param_name:
            existing[i] = export
            break
    else:
        existing.append(export)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Plot data saved → {output_path}")