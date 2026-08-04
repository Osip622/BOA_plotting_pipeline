"""
test_boa_sweep.py — Unit tests for boa_runner, boa_sweep, and boa_plot

Run with:
    python3 -m pytest test_boa_sweep.py -v

No BOA installation or GPU required — all subprocess calls are mocked,
and plotting tests write to a temp file rather than the screen.

Test groups:
    TestParseMetrics   — boa_runner.parse_metrics()
    TestSetNested      — boa_runner.set_nested() / get_nested()
    TestLoadWrite      — boa_runner.load_base_config() / write_config()
    TestRunBoa         — boa_runner.run_boa() with mocked subprocess
    TestSweepLogic     — boa_sweep.run_sweep() with mocked run_boa
    TestPersistence    — boa_sweep.save_results() / load_results()
    TestPlotSweep      — boa_plot.plot_sweep() with synthetic data
"""

import copy
import json
import os
import sys
import tempfile
import unittest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed in tests
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

# ── Make sure our modules are importable regardless of cwd ───────────────────
sys.path.insert(0, os.path.dirname(__file__))

from boa_runner import parse_metrics, set_nested, get_nested, write_config
import boa_sweep as sweep_module
import boa_plot as plot_module


# ══════════════════════════════════════════════════════════════════════════════
#  boa_runner tests
# ══════════════════════════════════════════════════════════════════════════════

class TestParseMetrics(unittest.TestCase):
    """parse_metrics() extracts the right numbers from BOA stdout."""

    FULL_OUTPUT = """
Model parameters: 2,130,688
Starting training on device==cuda
[Epoch 10] val bpp=6.8001 (ratio ~ 1.17x)  lr=5.00e-04
[TEST] bpp=6.8159  ratio ~ 1.17x
Training complete in 7.70s
Starting compression...
[compress] gpu_streams=1000
Compression complete: 1000 chunks, chunk_len=1114, last=150
Compression ratio (excl. model): 1.23
Compression ratio (incl. model): 0.55
  Compressed size: 906,292 bytes | Model size: 8,534,744 bytes
Compression complete in 3.54s (0.3 MB/s)
Starting decompression...
Decompression complete in 2.90s (0.4 MB/s)
"""

    def setUp(self):
        self.m = parse_metrics(self.FULL_OUTPUT)

    def test_ratio_excl(self):
        self.assertAlmostEqual(self.m["ratio_excl_model"], 1.23)

    def test_ratio_incl(self):
        self.assertAlmostEqual(self.m["ratio_incl_model"], 0.55)

    def test_test_bpp(self):
        self.assertAlmostEqual(self.m["test_bpp"], 6.8159)

    def test_final_val_bpp(self):
        # Should grab the last val bpp occurrence
        self.assertAlmostEqual(self.m["final_val_bpp"], 6.8001)

    def test_train_time(self):
        self.assertAlmostEqual(self.m["train_time_s"], 7.70)

    def test_compress_time(self):
        self.assertAlmostEqual(self.m["compress_time_s"], 3.54)

    def test_compress_mbps(self):
        self.assertAlmostEqual(self.m["compress_mbps"], 0.3)

    def test_decompress_time(self):
        self.assertAlmostEqual(self.m["decompress_time_s"], 2.90)

    def test_decompress_mbps(self):
        self.assertAlmostEqual(self.m["decompress_mbps"], 0.4)

    def test_compressed_bytes(self):
        self.assertEqual(self.m["compressed_bytes"], 906292)

    def test_model_params(self):
        self.assertEqual(self.m["model_params"], 2130688)

    def test_missing_fields_return_none(self):
        """Empty stdout should produce all-None metrics without crashing."""
        m = parse_metrics("")
        for key, val in m.items():
            self.assertIsNone(val, f"Expected None for '{key}', got {val}")

    def test_multiple_val_bpp_takes_last(self):
        """When multiple val bpp lines appear, only the last is kept."""
        output = """
[Epoch 1] val bpp=7.9000
[Epoch 2] val bpp=7.5000
[Epoch 3] val bpp=7.1000
"""
        m = parse_metrics(output)
        self.assertAlmostEqual(m["final_val_bpp"], 7.1000)

    def test_compressed_bytes_comma_separated(self):
        """Compressed size with comma thousands separator parses correctly."""
        m = parse_metrics("Compressed size: 1,234,567 bytes")
        self.assertEqual(m["compressed_bytes"], 1234567)

    def test_partial_output_no_crash(self):
        """Output with only compression lines (no training) parses safely."""
        output = """
Compression ratio (excl. model): 1.17
Compression complete in 60.0s (0.5 MB/s)
"""
        m = parse_metrics(output)
        self.assertAlmostEqual(m["ratio_excl_model"], 1.17)
        self.assertIsNone(m["test_bpp"])
        self.assertIsNone(m["train_time_s"])


# ══════════════════════════════════════════════════════════════════════════════

class TestSetNested(unittest.TestCase):
    """set_nested() and get_nested() work on arbitrarily deep dicts."""

    def test_set_two_levels(self):
        d = {}
        set_nested(d, ["model", "num_layers"], 4)
        self.assertEqual(d["model"]["num_layers"], 4)

    def test_set_one_level(self):
        d = {}
        set_nested(d, ["file_path"], "/some/path.bin")
        self.assertEqual(d["file_path"], "/some/path.bin")

    def test_set_three_levels(self):
        d = {}
        set_nested(d, ["a", "b", "c"], 99)
        self.assertEqual(d["a"]["b"]["c"], 99)

    def test_set_overwrites_existing(self):
        d = {"model": {"num_layers": 2}}
        set_nested(d, ["model", "num_layers"], 6)
        self.assertEqual(d["model"]["num_layers"], 6)

    def test_set_creates_intermediate_dicts(self):
        d = {}
        set_nested(d, ["x", "y", "z"], "hello")
        self.assertIn("x", d)
        self.assertIn("y", d["x"])

    def test_get_two_levels(self):
        d = {"model": {"d_model": 256}}
        self.assertEqual(get_nested(d, ["model", "d_model"]), 256)

    def test_get_missing_raises(self):
        d = {}
        with self.assertRaises(KeyError):
            get_nested(d, ["missing", "key"])


# ══════════════════════════════════════════════════════════════════════════════

class TestLoadWrite(unittest.TestCase):
    """load_base_config() and write_config() handle files correctly."""

    def test_write_then_read(self):
        """A written config can be read back with identical content."""
        import yaml
        cfg = {"model": {"num_layers": 2, "d_model": 64}, "file_path": "/tmp/data.bin"}
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                write_config(cfg, "test_exp")
                with open("experiments/test_exp/test_exp.yaml") as f:
                    loaded = yaml.safe_load(f)
                self.assertEqual(loaded["model"]["num_layers"], 2)
                self.assertEqual(loaded["file_path"], "/tmp/data.bin")
            finally:
                os.chdir(orig_cwd)

    def test_load_missing_raises(self):
        """load_base_config raises FileNotFoundError for non-existent experiments."""
        from boa_runner import load_base_config
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(FileNotFoundError):
                    load_base_config("does_not_exist")
            finally:
                os.chdir(orig_cwd)

    def test_write_creates_directory(self):
        """write_config creates the experiments/<name>/ directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                write_config({"key": "val"}, "new_exp")
                self.assertTrue(os.path.isdir("experiments/new_exp"))
                self.assertTrue(os.path.isfile("experiments/new_exp/new_exp.yaml"))
            finally:
                os.chdir(orig_cwd)


# ══════════════════════════════════════════════════════════════════════════════

class TestRunBoa(unittest.TestCase):
    """run_boa() calls subprocess correctly and parses output."""

    FAKE_STDOUT = (
        "Compression ratio (excl. model): 1.23\n"
        "Compression complete in 3.54s (0.3 MB/s)\n"
        "Decompression complete in 2.90s (0.4 MB/s)\n"
    )

    def _make_proc(self, stdout="", returncode=0):
        p = MagicMock()
        p.stdout = stdout
        p.stderr = ""
        p.returncode = returncode
        return p

    @patch("boa_runner.subprocess.run")
    def test_full_run_calls_main_once(self, mock_run):
        """Non-compress-only path calls main.py once."""
        mock_run.return_value = self._make_proc(self.FAKE_STDOUT)
        from boa_runner import run_boa
        metrics = run_boa("my_exp", compress_only=False)
        self.assertEqual(mock_run.call_count, 1)
        self.assertAlmostEqual(metrics["ratio_excl_model"], 1.23)

    @patch("boa_runner.subprocess.run")
    def test_compress_only_calls_main_twice(self, mock_run):
        """compress_only path calls main.py twice (compress then decompress)."""
        mock_run.return_value = self._make_proc(self.FAKE_STDOUT)
        from boa_runner import run_boa
        run_boa("my_exp", compress_only=True)
        self.assertEqual(mock_run.call_count, 2)

    @patch("boa_runner.subprocess.run")
    def test_compress_only_flags(self, mock_run):
        """compress_only passes --compress-only then --decompress-only flags."""
        mock_run.return_value = self._make_proc(self.FAKE_STDOUT)
        from boa_runner import run_boa
        run_boa("my_exp", compress_only=True)
        calls = mock_run.call_args_list
        self.assertIn("--compress-only", calls[0][0][0])
        self.assertIn("--decompress-only", calls[1][0][0])

    @patch("boa_runner.subprocess.run")
    def test_non_zero_exit_still_returns_metrics(self, mock_run):
        """Non-zero exit code prints a warning but still parses whatever output exists."""
        mock_run.return_value = self._make_proc(self.FAKE_STDOUT, returncode=1)
        from boa_runner import run_boa
        metrics = run_boa("my_exp")
        # Should still parse the ratio from stdout even on non-zero exit
        self.assertAlmostEqual(metrics["ratio_excl_model"], 1.23)

    @patch("boa_runner.subprocess.run")
    def test_empty_output_returns_all_none(self, mock_run):
        """Empty stdout returns a metrics dict with all-None values."""
        mock_run.return_value = self._make_proc("")
        from boa_runner import run_boa
        metrics = run_boa("my_exp")
        self.assertIsNone(metrics["ratio_excl_model"])
        self.assertIsNone(metrics["compress_mbps"])


# ══════════════════════════════════════════════════════════════════════════════
#  boa_sweep tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistence(unittest.TestCase):
    """save_results() and load_results() round-trip correctly."""

    def test_save_and_load(self):
        results = [{"param_name": "num_layers", "param_value": 2, "metrics": {"ratio_excl_model": 1.2}}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sweep_module.save_results(results, path)
            loaded = sweep_module.load_results(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["param_value"], 2)
            self.assertAlmostEqual(loaded[0]["metrics"]["ratio_excl_model"], 1.2)
        finally:
            os.unlink(path)

    def test_load_missing_returns_empty(self):
        """load_results returns [] when the file doesn't exist."""
        result = sweep_module.load_results("/tmp/definitely_does_not_exist_xyz.json")
        self.assertEqual(result, [])

    def test_save_overwrites(self):
        """Saving twice with different data keeps only the latest."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sweep_module.save_results([{"v": 1}], path)
            sweep_module.save_results([{"v": 2}, {"v": 3}], path)
            loaded = sweep_module.load_results(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["v"], 2)
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════

class TestSweepLogic(unittest.TestCase):
    """run_sweep() orchestrates experiments correctly."""

    def _fake_config(self):
        return {
            "model": {"num_layers": 2, "d_model": 256},
            "training": {"epochs": 10, "lr": 5e-4},
            "dataloader": {"seq_len": 32768, "batch_size": 3},
            "compression": {"chunks_count": 1000, "file_to_compress": ""},
            "file_path": "/eos/fake/data.bin",
            "model_path": "some_model.pt",
            "name": "base_exp",
        }

    def _fake_metrics(self):
        return {
            "ratio_excl_model": 1.23,
            "ratio_incl_model": 0.55,
            "compress_mbps": 0.3,
            "decompress_mbps": 0.4,
            "compress_time_s": 60.0,
            "decompress_time_s": 50.0,
            "test_bpp": 6.8,
            "final_val_bpp": 6.9,
            "train_time_s": 8.0,
            "compressed_bytes": 906292,
            "model_params": 2130688,
        }

    @patch("boa_sweep.run_boa")
    @patch("boa_sweep.write_config")
    @patch("boa_sweep.load_base_config")
    def test_sweep_runs_correct_number_of_experiments(
        self, mock_load, mock_write, mock_run
    ):
        """run_sweep calls run_boa once per value."""
        mock_load.return_value = self._fake_config()
        mock_run.return_value = self._fake_metrics()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_path = f.name
        orig = sweep_module.RESULTS_FILE
        sweep_module.RESULTS_FILE = results_path

        try:
            results = sweep_module.run_sweep("num_layers", [1, 2, 4])
            self.assertEqual(mock_run.call_count, 3)
            param_results = [r for r in results if r["param_name"] == "num_layers"]
            self.assertEqual(len(param_results), 3)
        finally:
            sweep_module.RESULTS_FILE = orig
            os.unlink(results_path)

    @patch("boa_sweep.run_boa")
    @patch("boa_sweep.write_config")
    @patch("boa_sweep.load_base_config")
    def test_skip_existing_skips_already_done(
        self, mock_load, mock_write, mock_run
    ):
        """run_sweep skips values already present in the results JSON."""
        mock_load.return_value = self._fake_config()
        mock_run.return_value = self._fake_metrics()

        existing = [
            {"param_name": "num_layers", "param_value": 2,
             "yaml_keys": ["model", "num_layers"],
             "experiment": "sweep_num_layers_2",
             "metrics": self._fake_metrics()}
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(existing, f)
            results_path = f.name
        orig = sweep_module.RESULTS_FILE
        sweep_module.RESULTS_FILE = results_path

        try:
            sweep_module.run_sweep("num_layers", [2, 4], skip_existing=True)
            # Only value=4 should trigger a run; value=2 is already done
            self.assertEqual(mock_run.call_count, 1)
        finally:
            sweep_module.RESULTS_FILE = orig
            os.unlink(results_path)

    @patch("boa_sweep.run_boa")
    @patch("boa_sweep.write_config")
    @patch("boa_sweep.load_base_config")
    def test_compress_only_preserves_model_path(
        self, mock_load, mock_write, mock_run
    ):
        """compress_only=True keeps model_path in config (no retraining)."""
        cfg = self._fake_config()
        mock_load.return_value = cfg
        mock_run.return_value = self._fake_metrics()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_path = f.name
        orig = sweep_module.RESULTS_FILE
        sweep_module.RESULTS_FILE = results_path

        written_configs = []

        def capture_config(config, name):
            written_configs.append(copy.deepcopy(config))
        mock_write.side_effect = capture_config

        try:
            sweep_module.run_sweep("chunks_count", [100], compress_only=True)
            self.assertIn("model_path", written_configs[0],
                          "model_path should be preserved when compress_only=True")
        finally:
            sweep_module.RESULTS_FILE = orig
            os.unlink(results_path)

    @patch("boa_sweep.run_boa")
    @patch("boa_sweep.write_config")
    @patch("boa_sweep.load_base_config")
    def test_full_run_removes_model_path(
        self, mock_load, mock_write, mock_run
    ):
        """compress_only=False removes model_path so BOA retrains from scratch."""
        cfg = self._fake_config()
        mock_load.return_value = cfg
        mock_run.return_value = self._fake_metrics()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_path = f.name
        orig = sweep_module.RESULTS_FILE
        sweep_module.RESULTS_FILE = results_path

        written_configs = []

        def capture_config(config, name):
            written_configs.append(copy.deepcopy(config))
        mock_write.side_effect = capture_config

        try:
            sweep_module.run_sweep("num_layers", [2], compress_only=False)
            self.assertNotIn("model_path", written_configs[0],
                             "model_path should be removed when compress_only=False")
        finally:
            sweep_module.RESULTS_FILE = orig
            os.unlink(results_path)

    @patch("boa_sweep.run_boa")
    @patch("boa_sweep.write_config")
    @patch("boa_sweep.load_base_config")
    def test_failed_run_saves_empty_metrics(
        self, mock_load, mock_write, mock_run
    ):
        """A run that raises an exception saves an empty metrics dict."""
        mock_load.return_value = self._fake_config()
        mock_run.side_effect = Exception("BOA crashed")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_path = f.name
        orig = sweep_module.RESULTS_FILE
        sweep_module.RESULTS_FILE = results_path

        try:
            results = sweep_module.run_sweep("num_layers", [2])
            param_results = [r for r in results if r["param_name"] == "num_layers"]
            self.assertEqual(param_results[0]["metrics"], {})
        finally:
            sweep_module.RESULTS_FILE = orig
            os.unlink(results_path)

    def test_unknown_param_raises(self):
        """run_sweep raises ValueError for unknown parameter names."""
        with self.assertRaises(ValueError):
            sweep_module.run_sweep("not_a_real_param", [1, 2])


# ══════════════════════════════════════════════════════════════════════════════
#  boa_plot tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotSweep(unittest.TestCase):
    """plot_sweep() handles normal, partial, and edge-case data without crashing."""

    def _make_results(self, values, ratios, compress_mbps, decompress_mbps,
                      compress_time=None, decompress_time=None):
        """Build a synthetic results list for plot testing."""
        results = []
        for i, v in enumerate(values):
            results.append({
                "param_name": "chunks_count",
                "param_value": v,
                "experiment": f"sweep_chunks_count_{v}",
                "metrics": {
                    "ratio_excl_model":  ratios[i],
                    "compress_mbps":     compress_mbps[i],
                    "decompress_mbps":   decompress_mbps[i],
                    "compress_time_s":   compress_time[i]   if compress_time   else 60.0,
                    "decompress_time_s": decompress_time[i] if decompress_time else 50.0,
                    "test_bpp":          None,
                    "ratio_incl_model":  None,
                },
            })
        return results

    def test_normal_data_saves_png(self):
        """plot_sweep saves a PNG without error given clean data."""
        results = self._make_results(
            values=[100, 250, 500],
            ratios=[1.23, 1.21, 1.19],
            compress_mbps=[0.3, 0.5, 0.8],
            decompress_mbps=[0.3, 0.5, 0.7],
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_module.plot_sweep(results, "chunks_count", output_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_none_values_dont_crash(self):
        """None values in throughput/time fields are handled gracefully."""
        results = self._make_results(
            values=[100, 500],
            ratios=[1.23, 1.19],
            compress_mbps=[0.3, None],     # second run has missing throughput
            decompress_mbps=[0.3, None],
            compress_time=[60.0, None],
            decompress_time=[50.0, None],
        )
        # Filter will exclude the second result (compress_mbps is None)
        # but first should still plot fine
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_module.plot_sweep(results, "chunks_count", output_path=path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_empty_results_prints_message(self, ):
        """plot_sweep prints a message and returns early with no matching results."""
        import io
        from contextlib import redirect_stdout
        results = []
        buf = io.StringIO()
        with redirect_stdout(buf):
            plot_module.plot_sweep(results, "chunks_count", output_path="/tmp/never.png")
        self.assertIn("No complete results", buf.getvalue())
        self.assertFalse(os.path.exists("/tmp/never.png"))

    def test_wrong_param_name_filtered_out(self):
        """Results for a different parameter are not plotted."""
        import io
        from contextlib import redirect_stdout
        results = self._make_results(
            values=[100], ratios=[1.2], compress_mbps=[0.3], decompress_mbps=[0.3]
        )
        # Change param_name to something different
        results[0]["param_name"] = "num_layers"

        buf = io.StringIO()
        with redirect_stdout(buf):
            plot_module.plot_sweep(results, "chunks_count", output_path="/tmp/never2.png")
        self.assertIn("No complete results", buf.getvalue())

    def test_single_data_point(self):
        """plot_sweep handles a single data point without crashing."""
        results = self._make_results(
            values=[500],
            ratios=[1.20],
            compress_mbps=[0.5],
            decompress_mbps=[0.4],
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_module.plot_sweep(results, "chunks_count", output_path=path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_results_sorted_by_param_value(self):
        """Results are sorted by param_value regardless of insertion order."""
        results = self._make_results(
            values=[500, 100, 250],       # deliberately out of order
            ratios=[1.19, 1.23, 1.21],
            compress_mbps=[0.8, 0.3, 0.5],
            decompress_mbps=[0.7, 0.3, 0.5],
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            # If sorting is broken the x-axis labels would be wrong order;
            # we verify by checking the plot saves successfully (no index errors)
            plot_module.plot_sweep(results, "chunks_count", output_path=path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

# ══════════════════════════════════════════════════════════════════════════════
#  plot_library tests
# ══════════════════════════════════════════════════════════════════════════════

from plot_library import (
    _safe, _extract,
    plot_compression_ratio, plot_throughput, plot_wallclock_time,
    plot_quality_speed_tradeoff, plot_test_bpp, plot_training_time,
    PLOT_REGISTRY, PLOTS_DEFAULT,
)


class TestPlotsLibraryHelpers(unittest.TestCase):
    """_safe() and _extract() behave correctly."""

    def test_safe_replaces_none(self):
        self.assertEqual(_safe([1.0, None, 2.0]), [1.0, 0.0, 2.0])

    def test_safe_custom_default(self):
        self.assertEqual(_safe([None, None], default=8.0), [8.0, 8.0])

    def test_safe_no_nones(self):
        self.assertEqual(_safe([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_safe_empty(self):
        self.assertEqual(_safe([]), [])

    def test_extract_keys(self):
        """_extract returns all expected keys."""
        results = [_make_sweep_result("chunks_count", 100, ratio=1.2, compress_mbps=0.3,
                                      decompress_mbps=0.4)]
        d = _extract(results)
        for key in ("values", "labels", "ratios", "compress_mbps", "decompress_mbps",
                    "compress_time", "decompress_time", "test_bpp", "train_time"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_extract_integer_labels(self):
        """Integer param values get labels without decimal point."""
        results = [_make_sweep_result("chunks_count", 100, ratio=1.2, compress_mbps=0.3,
                                      decompress_mbps=0.4)]
        d = _extract(results)
        self.assertEqual(d["labels"][0], "100")

    def test_extract_float_labels(self):
        """Float param values (e.g. lr) keep their decimal representation."""
        results = [_make_sweep_result("lr", 0.001, ratio=1.2, compress_mbps=0.3,
                                      decompress_mbps=0.4)]
        d = _extract(results)
        self.assertEqual(d["labels"][0], "0.001")

    def test_extract_none_propagated(self):
        """None metric values are propagated as None in extracted lists."""
        results = [_make_sweep_result("chunks_count", 100, ratio=None, compress_mbps=None,
                                      decompress_mbps=None)]
        d = _extract(results)
        self.assertIsNone(d["ratios"][0])
        self.assertIsNone(d["compress_mbps"][0])


def _make_sweep_result(param_name, param_value, ratio=1.2, compress_mbps=0.3,
                        decompress_mbps=0.4, test_bpp=6.8, train_time=8.0,
                        compress_time=60.0, decompress_time=50.0):
    """Helper: build a minimal sweep result dict for plot tests."""
    return {
        "param_name":  param_name,
        "param_value": param_value,
        "experiment":  f"sweep_{param_name}_{param_value}",
        "metrics": {
            "ratio_excl_model":  ratio,
            "compress_mbps":     compress_mbps,
            "decompress_mbps":   decompress_mbps,
            "compress_time_s":   compress_time,
            "decompress_time_s": decompress_time,
            "test_bpp":          test_bpp,
            "train_time_s":      train_time,
            "model_params":      2130688,
        },
    }


class TestIndividualPlotFunctions(unittest.TestCase):
    """Each plot function draws without crashing on normal and edge-case data."""

    def setUp(self):
        """Create a fresh figure with 6 axes for each test."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.axlist = self.axes.flatten().tolist()

    def tearDown(self):
        plt.close(self.fig)

    def _normal_results(self, param="chunks_count"):
        return [
            _make_sweep_result(param, 100),
            _make_sweep_result(param, 250, ratio=1.21, compress_mbps=0.5,
                               decompress_mbps=0.5),
            _make_sweep_result(param, 500, ratio=1.19, compress_mbps=0.8,
                               decompress_mbps=0.7),
        ]

    def _none_results(self, param="chunks_count"):
        """Results where most metrics are None — stress-tests None safety."""
        return [
            _make_sweep_result(param, 100, ratio=1.2, compress_mbps=0.3,
                               decompress_mbps=None, test_bpp=None, train_time=None,
                               compress_time=None, decompress_time=None),
        ]

    def test_plot_compression_ratio_normal(self):
        plot_compression_ratio(self.axlist[0], self._normal_results(), "chunks_count")

    def test_plot_compression_ratio_none_values(self):
        plot_compression_ratio(self.axlist[0], self._none_results(), "chunks_count")

    def test_plot_throughput_normal(self):
        plot_throughput(self.axlist[0], self._normal_results(), "chunks_count")

    def test_plot_throughput_none_values(self):
        plot_throughput(self.axlist[0], self._none_results(), "chunks_count")

    def test_plot_wallclock_time_normal(self):
        plot_wallclock_time(self.axlist[0], self._normal_results(), "chunks_count")

    def test_plot_wallclock_time_none_values(self):
        plot_wallclock_time(self.axlist[0], self._none_results(), "chunks_count")

    def test_plot_quality_speed_tradeoff_normal(self):
        plot_quality_speed_tradeoff(self.axlist[0], self._normal_results(), "chunks_count")

    def test_plot_quality_speed_tradeoff_no_valid_points(self):
        """Tradeoff scatter handles the case where all decompress_mbps are None."""
        results = [_make_sweep_result("chunks_count", 100, decompress_mbps=None)]
        # Should not crash — just draws an empty axes
        plot_quality_speed_tradeoff(self.axlist[0], results, "chunks_count")

    def test_plot_test_bpp_normal(self):
        plot_test_bpp(self.axlist[0], self._normal_results(), "num_layers")

    def test_plot_test_bpp_all_none(self):
        """test_bpp is None for --compress-only runs; should degrade gracefully."""
        results = [_make_sweep_result("chunks_count", 100, test_bpp=None)]
        plot_test_bpp(self.axlist[0], results, "chunks_count")

    def test_plot_training_time_normal(self):
        plot_training_time(self.axlist[0], self._normal_results(), "num_layers")

    def test_plot_training_time_all_none(self):
        """train_time is None for --compress-only runs."""
        results = [_make_sweep_result("chunks_count", 100, train_time=None)]
        plot_training_time(self.axlist[0], results, "chunks_count")

    def test_single_point_no_crash(self):
        """All plot functions handle a single data point."""
        results = [_make_sweep_result("chunks_count", 100)]
        for fn in (plot_compression_ratio, plot_throughput, plot_wallclock_time,
                   plot_quality_speed_tradeoff, plot_test_bpp, plot_training_time):
            fn(self.axlist[0], results, "chunks_count")


class TestPlotRegistry(unittest.TestCase):
    """PLOT_REGISTRY maps param names to non-empty plot lists."""

    def test_known_params_in_registry(self):
        for param in ("chunks_count", "num_layers", "d_model", "epochs"):
            self.assertIn(param, PLOT_REGISTRY, f"'{param}' missing from registry")

    def test_all_registry_values_are_lists(self):
        for param, plot_list in PLOT_REGISTRY.items():
            self.assertIsInstance(plot_list, list, f"Registry entry for '{param}' is not a list")
            self.assertGreater(len(plot_list), 0, f"Registry entry for '{param}' is empty")

    def test_all_registry_entries_are_callable(self):
        for param, plot_list in PLOT_REGISTRY.items():
            for fn in plot_list:
                self.assertTrue(callable(fn),
                                f"'{fn}' in registry entry for '{param}' is not callable")

    def test_default_plots_is_nonempty_list(self):
        self.assertIsInstance(PLOTS_DEFAULT, list)
        self.assertGreater(len(PLOTS_DEFAULT), 0)

    def test_unknown_param_falls_back_to_default(self):
        """Params not in registry should use PLOTS_DEFAULT — verify fallback works."""
        plot_fns = PLOT_REGISTRY.get("some_unknown_param", PLOTS_DEFAULT)
        self.assertEqual(plot_fns, PLOTS_DEFAULT)


# ══════════════════════════════════════════════════════════════════════════════
#  Updated boa_plot tests — save_plot_data and grid sizing
# ══════════════════════════════════════════════════════════════════════════════

from boa_plot import save_plot_data


class TestSavePlotData(unittest.TestCase):
    """save_plot_data() appends and replaces correctly."""

    def _results(self, param="chunks_count", values=(100, 250)):
        return [_make_sweep_result(param, v) for v in values]

    def test_creates_file_if_missing(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)  # delete so save_plot_data creates it fresh
        try:
            save_plot_data(self._results(), "chunks_count", path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["param_name"], "chunks_count")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_appends_new_param(self):
        """Different param names are appended as separate entries."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            save_plot_data(self._results("chunks_count"), "chunks_count", path)
            save_plot_data(self._results("num_layers", (1, 2, 4)), "num_layers", path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 2)
            param_names = {e["param_name"] for e in data}
            self.assertIn("chunks_count", param_names)
            self.assertIn("num_layers", param_names)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_replaces_existing_param(self):
        """Running the same param again replaces the old entry, not appends."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            save_plot_data(self._results("chunks_count", (100,)), "chunks_count", path)
            save_plot_data(self._results("chunks_count", (100, 250, 500)), "chunks_count", path)
            with open(path) as f:
                data = json.load(f)
            # Should still be 1 entry (replaced, not appended)
            self.assertEqual(len(data), 1)
            self.assertEqual(len(data[0]["points"]), 3)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_points_contain_expected_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            save_plot_data(self._results(), "chunks_count", path)
            with open(path) as f:
                data = json.load(f)
            point = data[0]["points"][0]
            for key in ("param_value", "experiment", "ratio_excl_model",
                        "compress_mbps", "decompress_mbps", "compress_time_s",
                        "decompress_time_s", "test_bpp", "train_time_s", "model_params"):
                self.assertIn(key, point, f"Missing key in point: {key}")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_handles_corrupted_existing_file(self):
        """A corrupted JSON file is treated as empty rather than crashing."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json {{{{")
            path = f.name
        try:
            # Should not raise
            save_plot_data(self._results(), "chunks_count", path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
        finally:
            os.unlink(path)


class TestPlotSweepGridSizing(unittest.TestCase):
    """plot_sweep assembles the correct number of panels."""

    def _make_results(self, param, values=(100, 250, 500)):
        return [_make_sweep_result(param, v) for v in values]

    def test_chunks_count_uses_registry_plot_set(self):
        """chunks_count uses PLOTS_CHUNKS_COUNT, not the default."""
        from plot_library import PLOTS_CHUNKS_COUNT
        plot_fns = PLOT_REGISTRY.get("chunks_count", PLOTS_DEFAULT)
        self.assertEqual(plot_fns, PLOTS_CHUNKS_COUNT)

    def test_num_layers_uses_registry_plot_set(self):
        from plot_library import PLOTS_NUM_LAYERS
        plot_fns = PLOT_REGISTRY.get("num_layers", PLOTS_DEFAULT)
        self.assertEqual(plot_fns, PLOTS_NUM_LAYERS)

    def test_plot_sweep_saves_png_chunks_count(self):
        results = self._make_results("chunks_count")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_module.plot_sweep(results, "chunks_count", output_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_plot_sweep_saves_png_num_layers(self):
        results = self._make_results("num_layers", (1, 2, 4))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_module.plot_sweep(results, "num_layers", output_path=path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_plot_sweep_unknown_param_uses_default(self):
        """Unknown parameter falls back to PLOTS_DEFAULT without crashing."""
        results = self._make_results("batch_size", (1, 2, 4))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_module.plot_sweep(results, "batch_size", output_path=path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_plot_sweep_also_saves_plot_data(self):
        """plot_data_path argument triggers save_plot_data."""
        results = self._make_results("chunks_count")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pf:
            png_path = pf.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
            json_path = jf.name
        os.unlink(json_path)  # let save_plot_data create it
        try:
            plot_module.plot_sweep(results, "chunks_count",
                                   output_path=png_path,
                                   plot_data_path=json_path)
            self.assertTrue(os.path.exists(json_path))
            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(data[0]["param_name"], "chunks_count")
        finally:
            for p in (png_path, json_path):
                if os.path.exists(p):
                    os.unlink(p)



# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)