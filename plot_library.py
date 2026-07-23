"""
boa_plots_library.py — Individual plot definitions for BOA sweep results.

Each function takes a matplotlib Axes object and the sweep results,
and draws one panel. Add a new function here to add a new plot —
the assembler in boa_plot.py picks them all up automatically.

To add a plot:
    1. Define a function with signature:
           def plot_my_thing(ax, sweep_results, param_name): ...
    2. Add it to PLOTS at the bottom of this file.
    3. Done — it appears in the next run automatically.

Default plot sets:
    PLOTS_CHUNKS_COUNT  — for sweeping chunks_count (throughput focus)
    PLOTS_NUM_LAYERS    — for sweeping num_layers (quality vs cost focus)
    PLOTS_DEFAULT       — fallback for any other parameter
"""

import numpy as np

def _safe(values: list, default: float = 0.0) -> list:
    """Replace None with default so matplotlib doesn't crash."""
    return [v if v is not None else default for v in values]


def _extract(sweep_results: list) -> dict:
    """Pull all metric lists out of sweep_results in one place."""
    return {
        "values":          [r["param_value"]                           for r in sweep_results],
        "labels":          [str(int(r["param_value"])) if float(r["param_value"]).is_integer()
                            else str(r["param_value"])                 for r in sweep_results],
        "ratios":          [r["metrics"].get("ratio_excl_model")       for r in sweep_results],
        "ratios_incl":          [r["metrics"].get("ratio_incl_model")       for r in sweep_results],
        "compress_mbps":   [r["metrics"].get("compress_mbps")          for r in sweep_results],
        "decompress_mbps": [r["metrics"].get("decompress_mbps")        for r in sweep_results],
        "compress_time":   [r["metrics"].get("compress_time_s")        for r in sweep_results],
        "decompress_time": [r["metrics"].get("decompress_time_s")      for r in sweep_results],
        "test_bpp":        [r["metrics"].get("test_bpp")               for r in sweep_results],
        "train_time":      [r["metrics"].get("train_time_s")           for r in sweep_results],
    }


# ── Individual plot functions ─────────────────────────────────────────────────
# Each takes (ax, sweep_results, param_name) and draws one panel.

def plot_compression_ratio(ax, sweep_results, param_name):
    """Bar chart of compression ratio (excl. model) vs parameter.

    Interesting for chunks_count: ratio should be roughly flat if chunk
    sizes are large enough. A big drop signals chunks are too small.
    Interesting for num_layers: expect diminishing returns after 2-4 layers.
    """
    d = _extract(sweep_results)
    x = np.arange(len(d["values"]))
    bars = ax.bar(x, _safe(d["ratios"]), color="#1baf7a", width=0.6)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No compression (1.0x)")
    ax.set_xticks(x); ax.set_xticklabels(d["labels"])
    ax.set_xlabel(param_name); ax.set_ylabel("Compression ratio (excl. model)")
    ax.set_title("Compression ratio\n(higher = better)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, d["ratios"]):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)



def plot_throughput(ax, sweep_results, param_name):
    """Line chart of compress + decompress throughput (MB/s) vs parameter.

    Interesting for chunks_count: more chunks = more parallel GPU streams =
    higher throughput, up to VRAM saturation.
    Decompression throughput is the critical number for HEP — compress once,
    decompress many times.
    """
    d = _extract(sweep_results)
    x = np.arange(len(d["values"]))
    ax.plot(x, _safe(d["compress_mbps"]),   marker="o", color="#4a90d9",
            linewidth=2, markersize=8, label="Compression MB/s")
    ax.plot(x, _safe(d["decompress_mbps"]), marker="s", color="#e07b39",
            linewidth=2, markersize=8, label="Decompression MB/s")
    ax.set_xticks(x); ax.set_xticklabels(d["labels"])
    ax.set_xlabel(param_name); ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Throughput\n(higher = faster)")
    ax.legend(); ax.grid(True, alpha=0.3)
    for i, (c, dv) in enumerate(zip(d["compress_mbps"], d["decompress_mbps"])):
        if c is not None:
            ax.annotate(f"{c:.2f}", (x[i], c), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#4a90d9")
        if dv is not None:
            ax.annotate(f"{dv:.2f}", (x[i], dv), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#e07b39")


def plot_wallclock_time(ax, sweep_results, param_name):
    """Grouped bar chart of compress + decompress wall-clock time (seconds).

    Makes the cost concrete — easier to communicate than MB/s for
    estimating real-world cost at scale (e.g. processing a full ATLAS run).
    """
    d = _extract(sweep_results)
    x = np.arange(len(d["values"]))
    w = 0.35
    ax.bar(x - w/2, _safe(d["compress_time"]),   width=w, color="#4a90d9", label="Compression (s)")
    ax.bar(x + w/2, _safe(d["decompress_time"]), width=w, color="#e07b39", label="Decompression (s)")
    ax.set_xticks(x); ax.set_xticklabels(d["labels"])
    ax.set_xlabel(param_name); ax.set_ylabel("Time (seconds)")
    ax.set_title("Wall-clock time\n(lower = faster)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")


def plot_quality_speed_tradeoff(ax, sweep_results, param_name):
    """Scatter: compression ratio vs decompression throughput (Pareto tradeoff).

    Top-right = best. The Pareto frontier shows configs where you can't
    improve ratio without hurting speed, or vice versa.
    For HEP: decompression throughput on x-axis since data is read many times.
    """
    import matplotlib.pyplot as plt
    d = _extract(sweep_results)
    valid = [(r, dv, v) for r, dv, v in
             zip(d["ratios"], d["decompress_mbps"], d["values"])
             if r is not None and dv is not None]
    if valid:
        rs, dvs, vs = zip(*valid)
        scatter = ax.scatter(dvs, rs, c=range(len(rs)), cmap="viridis", s=120, zorder=5)
        for r, dv, v in zip(rs, dvs, vs):
            label = str(int(v)) if float(v).is_integer() else str(v)
            ax.annotate(f"n={label}", (dv, r),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label=f"{param_name} (lighter = larger)")
    ax.set_xlabel("Decompression throughput (MB/s)")
    ax.set_ylabel("Compression ratio (excl. model)")
    ax.set_title("Quality vs Speed tradeoff\n(top-right = best)")
    ax.grid(True, alpha=0.3)


def plot_test_bpp(ax, sweep_results, param_name):
    """Bar chart of test bpp vs parameter.

    Only populated for full train+compress runs (not --compress-only).
    Interesting for num_layers, d_model, epochs — shows whether more
    capacity actually improves the model's byte predictions.
    8.0 bpp = random (no compression). Lower = model learned more structure.
    """
    d = _extract(sweep_results)
    x = np.arange(len(d["values"]))
    bars = ax.bar(x, _safe(d["test_bpp"], default=8.0), color="#9b59b6", width=0.6)
    ax.axhline(8.0, color="gray", linestyle="--", linewidth=1, label="Random (8.0 bpp)")
    ax.set_xticks(x); ax.set_xticklabels(d["labels"])
    ax.set_xlabel(param_name); ax.set_ylabel("Bits per byte (test set)")
    ax.set_title("Test bpp\n(lower = better model)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, d["test_bpp"]):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)


def plot_training_time(ax, sweep_results, param_name):
    """Line chart of training time vs parameter.

    Only populated for full train+compress runs.
    Useful for num_layers and d_model to show the cost of more capacity.
    """
    d = _extract(sweep_results)
    x = np.arange(len(d["values"]))
    ax.plot(x, _safe(d["train_time"]), marker="o", color="#c0392b",
            linewidth=2, markersize=8)
    ax.set_xticks(x); ax.set_xticklabels(d["labels"])
    ax.set_xlabel(param_name); ax.set_ylabel("Training time (seconds)")
    ax.set_title("Training time\n(lower = cheaper)")
    ax.grid(True, alpha=0.3)
    for i, val in enumerate(d["train_time"]):
        if val is not None:
            ax.annotate(f"{val:.1f}s", (x[i], val),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)


# ── Plot sets ─────────────────────────────────────────────────────────────────
# Each is a list of plot functions to use for a given sweep parameter.
# The assembler in boa_plot.py lays these out automatically into a grid.

# chunks_count: throughput is the story — ratio should be flat, speed varies
PLOTS_CHUNKS_COUNT = [
    plot_compression_ratio,
    plot_throughput,
    plot_wallclock_time,
    plot_quality_speed_tradeoff,
]

# num_layers: quality vs training cost — bpp and train time are the story
PLOTS_NUM_LAYERS = [
    plot_compression_ratio,
    plot_test_bpp,
    plot_training_time,
    plot_quality_speed_tradeoff,
]

# d_model: same shape as num_layers — capacity vs cost
PLOTS_D_MODEL = [
    plot_compression_ratio,
    plot_test_bpp,
    plot_training_time,
    plot_throughput,
]

# epochs: convergence story — does more training actually help?
PLOTS_EPOCHS = [
    plot_test_bpp,
    plot_compression_ratio,
    plot_training_time,
    plot_quality_speed_tradeoff,
]

# Default fallback — shows all the core metrics
PLOTS_DEFAULT = [
    plot_compression_ratio,
    plot_throughput,
    plot_wallclock_time,
    plot_quality_speed_tradeoff,
]

# Registry — maps param name to its plot set
PLOT_REGISTRY = {
    "chunks_count": PLOTS_CHUNKS_COUNT,
    "num_layers":   PLOTS_NUM_LAYERS,
    "d_model":      PLOTS_D_MODEL,
    "epochs":       PLOTS_EPOCHS,
}
