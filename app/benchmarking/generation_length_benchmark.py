#!/usr/bin/env python3
"""
Simple driver to sweep generation length (frames) and call app/benchmarking/benchmark.py
to produce per-request latency CSVs annotated with generation_length.

Plots and deeper analysis are handled by the eval repository.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


class GenerationLengthSweep:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config["output_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.results_dir / "raw_data"
        self.raw_dir.mkdir(exist_ok=True)

    def _run_single(self, generation_length: int) -> Path | None:
        """Run app/benchmarking/benchmark.py once for a given generation_length."""
        csv_path = self.raw_dir / f"gen_length_{generation_length}.csv"

        cmd = [
            sys.executable,
            "app/benchmarking/benchmark.py",
            "--server_url",
            self.config["server_url"],
            "--num_requests",
            str(self.config["requests_per_length"]),
            "--output_file",
            str(csv_path),
            "--generation_length_frames",
            str(generation_length),
        ]

        print(f"\n=== GL={generation_length} frames ===")
        print("Running:", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config["timeout_seconds"]
            )
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout for GL={generation_length}")
            return None

        if result.returncode != 0:
            print(f"❌ Benchmark failed for GL={generation_length}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None

        if not csv_path.exists():
            print(f"❌ Expected CSV not found for GL={generation_length}: {csv_path}")
            return None

        # Attach generation_length column for downstream analysis
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                print(f"⚠️ Empty CSV for GL={generation_length}")
                return None
            df["generation_length"] = generation_length
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"⚠️ Failed to post-process CSV for GL={generation_length}: {e}")
            return None

        print(f"✓ Saved {len(df)} rows to {csv_path}")
        return csv_path

    def run(self) -> List[Path]:
        """Run sweep over configured generation_lengths."""
        gls = self.config["generation_lengths"]
        print("Generation length sweep:", gls)
        print("Requests per length:", self.config["requests_per_length"])
        print("Results under:", self.results_dir)

        csv_paths: List[Path] = []
        for gl in gls:
            path = self._run_single(gl)
            if path is not None:
                csv_paths.append(path)
            time.sleep(1.0)

        if not csv_paths:
            print("❌ No successful runs.")
        else:
            print("Completed sweep for GLs:", [p.stem for p in csv_paths])

        return csv_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep generation_length_frames and call app/benchmarking/benchmark.py to "
            "produce latency CSVs; analysis is done in eval."
        )
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8988/generate_accompaniment",
        help="StreamMUSE server URL (generate_accompaniment endpoint).",
    )
    parser.add_argument(
        "--generation_lengths",
        type=str,
        default="3,5,9,15",
        help="Comma-separated generation lengths (frames) to test, e.g. '3,5,9,15'.",
    )
    parser.add_argument(
        "--requests_per_length",
        type=int,
        default=100,
        help="Number of benchmark requests per generation length.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bench_results/generation_length_sweep",
        help="Directory to store raw CSV results.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=600,
        help="Timeout (seconds) for each app/benchmarking/benchmark.py run.",
    )

    args = parser.parse_args()

    try:
        gls = [int(x.strip()) for x in args.generation_lengths.split(",") if x.strip()]
    except ValueError:
        print("Error: invalid --generation_lengths; expected comma-separated integers.")
        return 1

    if not gls:
        print("Error: no generation lengths provided.")
        return 1

    if args.requests_per_length < 1:
        print("Error: --requests_per_length must be >= 1.")
        return 1

    config: Dict[str, Any] = {
        "server_url": args.server_url,
        "generation_lengths": gls,
        "requests_per_length": args.requests_per_length,
        "output_dir": args.output_dir,
        "timeout_seconds": args.timeout_seconds,
    }

    sweep = GenerationLengthSweep(config)
    sweep.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Enhanced benchmark script for analyzing generation length effects on StreamMUSE latency.

This script works with the existing server by running multiple benchmark sessions
with different GENERATION_LENGTH_FRAMES environment variable settings.
It does not modify any existing application files.
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import json
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statistics


class GenerationLengthBenchmark:
    """
    Manages generation length parameter sweep benchmarks by coordinating
    multiple runs of the existing benchmark.py script with different server configurations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config["output_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        self.raw_data_dir = self.results_dir / "raw_data"
        self.analysis_dir = self.results_dir / "analysis"
        self.plots_dir = self.results_dir / "plots"

        self.raw_data_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        self.all_results = []
        self.summary_stats = []

    def run_benchmark_for_generation_length(self, generation_length: int) -> bool:
        """
        Run benchmark for a specific generation length by starting server
        with appropriate environment variables and running existing benchmark.py
        """
        print("\n" + "=" * 60)
        print(f"Testing Generation Length: {generation_length} frames")
        print("=" * 60)

        # Define output files for this generation length
        csv_file = self.raw_data_dir / f"gen_length_{generation_length}.csv"

        # Prepare benchmark command
        benchmark_cmd = [
            sys.executable,
            "app/benchmarking/benchmark.py",
            "--server_url",
            self.config["server_url"],
            "--num_requests",
            str(self.config["requests_per_length"]),
            "--output_file",
            str(csv_file),
            "--generation_length_frames",
            str(generation_length),
        ]

        print(f"Running benchmark with {self.config['requests_per_length']} requests...")
        print(f"Command: {' '.join(benchmark_cmd)}")

        try:
            # Run the existing benchmark script
            result = subprocess.run(
                benchmark_cmd,
                capture_output=True,
                text=True,
                timeout=self.config["timeout_seconds"],
            )

            if result.returncode != 0:
                print(
                    f"❌ Benchmark failed for generation length {generation_length}"
                )
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False

            print(f"✅ Benchmark completed for generation length {generation_length}")

            # Verify output files exist
            if not csv_file.exists():
                print(f"❌ CSV output file not found: {csv_file}")
                return False

            # Load and validate the results
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                print(f"❌ No data in CSV file for generation length {generation_length}")
                return False

            print(f"📊 Collected {len(df)} successful requests")

            # Add generation length column to the data
            df["generation_length"] = generation_length

            # Re-save with generation length column
            df.to_csv(csv_file, index=False)

            # Add to our combined results
            self.all_results.append(df)

            # Calculate summary statistics
            summary = self._calculate_summary_stats(df, generation_length)
            self.summary_stats.append(summary)

            return True

        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timed out for generation length {generation_length}")
            return False
        except Exception as e:
            print(
                f"❌ Error running benchmark for generation length {generation_length}: {e}"
            )
            return False

    def _calculate_summary_stats(
        self, df: pd.DataFrame, generation_length: int
    ) -> Dict[str, Any]:
        """Calculate summary statistics for a single generation length."""

        def safe_stats(series):
            """Calculate stats safely, handling empty series."""
            if len(series) == 0:
                return {
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0,
                    "median": 0,
                    "p95": 0,
                    "p99": 0,
                }
            return {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "p95": series.quantile(0.95),
                "p99": series.quantile(0.99),
            }

        summary = {
            "generation_length": generation_length,
            "num_requests": len(df),
            "num_successful_requests": len(df),  # All rows in CSV are successful
        }

        # Add statistics for each timing metric
        metrics = [
            "round_trip_time",
            "server_processing_duration",
            "inference_duration",
            "preprocess_duration",
            "postprocess_duration",
            "total_network_latency",
        ]

        for metric in metrics:
            if metric in df.columns:
                stats = safe_stats(df[metric])
                for stat_name, value in stats.items():
                    summary[f"{metric}_{stat_name}"] = value

        # Add notes generated stats
        if "num_generated_notes" in df.columns:
            notes_stats = safe_stats(df["num_generated_notes"])
            for stat_name, value in notes_stats.items():
                summary[f"num_generated_notes_{stat_name}"] = value

        return summary

    def run_parameter_sweep(self) -> bool:
        """
        Run the complete parameter sweep across all specified generation lengths.
        Note: This requires manual server restart between generation lengths.
        """
        print("🚀 Starting Generation Length Parameter Sweep")
        print(f"Testing generation lengths: {self.config['generation_lengths']}")
        print(f"Requests per length: {self.config['requests_per_length']}")
        print(f"Output directory: {self.results_dir}")

        if not self.config["auto_server_restart"]:
            print("\n⚠️  MANUAL MODE: You need to restart the server with different")
            print("   GENERATION_LENGTH_FRAMES values between each test.")
            print("   The script will pause and wait for your confirmation.")

        successful_runs = 0
        total_runs = len(self.config["generation_lengths"])

        for i, gen_length in enumerate(self.config["generation_lengths"]):
            if not self.config["auto_server_restart"]:
                if i > 0:  # Skip prompt for first run
                    print("\n⏸️  Please restart the server with:")
                    print(
                        "   GENERATION_LENGTH_FRAMES="
                        f"{gen_length} uvicorn app.server:app "
                        f"--host 0.0.0.0 --port {self.config['server_port']}"
                    )
                    print("   Then press Enter to continue...")
                    input()
                else:
                    print(
                        "\n🔧 Please ensure server is running with "
                        f"GENERATION_LENGTH_FRAMES={gen_length}"
                    )
                    print("   Press Enter when ready...")
                    input()

            # Test server connection
            if not self._test_server_connection():
                print(
                    f"❌ Cannot connect to server. Skipping generation length {gen_length}"
                )
                continue

            # Run benchmark for this generation length
            if self.run_benchmark_for_generation_length(gen_length):
                successful_runs += 1
            else:
                print(
                    f"⚠️  Failed to complete benchmark for generation length {gen_length}"
                )

            # Small delay between runs
            time.sleep(2)

        print("\n🏁 Parameter sweep completed!")
        print(f"   Successful runs: {successful_runs}/{total_runs}")

        if successful_runs > 0:
            self._export_combined_results()
            if self.config["generate_plots"]:
                self._generate_visualizations()
            if self.config["generate_report"]:
                self._generate_analysis_report()

        return successful_runs > 0

    def _test_server_connection(self) -> bool:
        """Test if server is responding."""
        import requests

        try:
            # Test with a simple request to clear_history endpoint
            clear_url = self.config["server_url"].replace(
                "/generate_accompaniment", "/clear_history"
            )
            response = requests.post(clear_url, timeout=5)
            return response.status_code in [200, 503]  # 503 is OK (engine not loaded)
        except Exception:
            return False

    def _export_combined_results(self):
        """Export combined results to CSV and summary statistics."""
        print("\n📁 Exporting combined results...")

        # Combine all individual results
        if self.all_results:
            combined_df = pd.concat(self.all_results, ignore_index=True)
            combined_csv = (
                self.analysis_dir / "detailed_results_all_generation_lengths.csv"
            )
            combined_df.to_csv(combined_csv, index=False)
            print(f"   Detailed results: {combined_csv}")

        # Export summary
        if self.summary_stats:
            summary_df = pd.DataFrame(self.summary_stats)
            summary_csv = self.analysis_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"   Summary statistics: {summary_csv}")

    def _generate_visualizations(self):
        """Generate visualization plots for the benchmark results."""
        print("\n📊 Generating visualizations...")

        # Load summary data
        summary_file = self.analysis_dir / "summary_statistics.csv"
        if not summary_file.exists():
            print("❌ No summary_statistics.csv found; cannot generate plots.")
            return

        summary_df = pd.read_csv(summary_file)

        # Plot round trip time vs generation length
        plt.figure(figsize=(10, 6))
        plt.plot(
            summary_df["generation_length"],
            summary_df["round_trip_time_mean"] * 1000,
            "o-",
            label="Mean RTT",
        )
        plt.fill_between(
            summary_df["generation_length"],
            (summary_df["round_trip_time_mean"] - summary_df["round_trip_time_std"])
            * 1000,
            (summary_df["round_trip_time_mean"] + summary_df["round_trip_time_std"])
            * 1000,
            alpha=0.2,
            label="±1 std dev",
        )
        plt.xlabel("Generation Length (Frames)")
        plt.ylabel("Round Trip Time (ms)")
        plt.title("Round Trip Time vs Generation Length")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "round_trip_time_vs_generation_length.png", dpi=300)
        plt.close()

        # Additional plots can be added here following similar patterns

    def _generate_analysis_report(self):
        """Generate automated analysis report."""
        print("\n📝 Generating analysis report...")

        if not self.summary_stats:
            return

        summary_df = pd.DataFrame(self.summary_stats)

        report_path = self.analysis_dir / "benchmark_analysis_report.md"

        with open(report_path, "w") as f:
            f.write("# Generation Length Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(
                f"- **Generation Lengths Tested:** "
                f"{list(summary_df['generation_length'])}\n"
            )
            f.write(
                f"- **Total Requests:** {summary_df['num_requests'].sum()}\n"
            )
            f.write(
                f"- **Successful Requests:** "
                f"{summary_df['num_successful_requests'].sum()}\n\n"
            )

            # Key findings
            f.write("## Key Findings\n\n")

            # Find optimal generation length for latency
            min_latency_idx = summary_df["round_trip_time_mean"].idxmin()
            optimal_gen_length = summary_df.iloc[min_latency_idx]["generation_length"]
            min_latency = (
                summary_df.iloc[min_latency_idx]["round_trip_time_mean"] * 1000
            )

            f.write(
                f"- **Optimal Generation Length for Latency:** {optimal_gen_length} "
                f"frames ({min_latency:.1f}ms mean round trip)\n"
            )

            # Latency range
            min_rtt = summary_df["round_trip_time_mean"].min() * 1000
            max_rtt = summary_df["round_trip_time_mean"].max() * 1000
            f.write(f"- **Latency Range:** {min_rtt:.1f}ms - {max_rtt:.1f}ms\n")

            # Variability analysis
            min_var_idx = summary_df["round_trip_time_std"].idxmin()
            most_consistent = summary_df.iloc[min_var_idx]["generation_length"]
            min_std = (
                summary_df.iloc[min_var_idx]["round_trip_time_std"] * 1000
            )
            f.write(
                f"- **Most Consistent Performance:** {most_consistent} frames "
                f"({min_std:.1f}ms std dev)\n\n"
            )

            # Performance table
            f.write("## Performance Table\n\n")
            f.write(
                "| Gen Length | Mean RTT (ms) | Std RTT (ms) | "
                "Mean Inference (ms) | Notes Generated |\n"
            )
            f.write(
                "|------------|---------------|--------------|"
                "---------------------|------------------|\n"
            )

            for _, row in summary_df.iterrows():
                gen_len = int(row["generation_length"])
                mean_rtt = row["round_trip_time_mean"] * 1000
                std_rtt = row["round_trip_time_std"] * 1000
                mean_inf = row["inference_duration_mean"] * 1000
                notes = row.get("num_generated_notes_mean", 0)

                f.write(
                    f"| {gen_len} | {mean_rtt:.1f} | {std_rtt:.1f} | "
                    f"{mean_inf:.1f} | {notes:.1f} |\n"
                )

            f.write("\n## Data Files\n\n")
            f.write(
                "- **Detailed Results:** "
                "`detailed_results_all_generation_lengths.csv`\n"
            )
            f.write(
                "- **Summary Statistics:** "
                "`summary_statistics.csv`\n"
            )
            f.write("- **Raw Data:** `raw_data/` directory\n")
            f.write("- **Visualizations:** `plots/` directory\n")

        print(f"   Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Enhanced benchmark for analyzing generation length effects "
            "on StreamMUSE latency"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --generation_lengths 5,10,15,20,25,30 --output_dir results/gen_length_study
  %(prog)s --generation_lengths 10,20,30 --requests_per_length 50 --generate_plots
        """,
    )

    # Core parameters
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000/generate_accompaniment",
        help="StreamMUSE server URL",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8000,
        help="Server port (for restart instructions)",
    )
    parser.add_argument(
        "--generation_lengths",
        type=str,
        default="5,10,15,20,25,30",
        help="Comma-separated generation lengths to test",
    )
    parser.add_argument(
        "--requests_per_length",
        type=int,
        default=50,
        help="Number of requests per generation length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/generation_length_analysis",
        help="Output directory for results",
    )

    # Control options
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate analysis report",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=300,
        help="Timeout for each benchmark run",
    )
    parser.add_argument(
        "--auto_server_restart",
        action="store_true",
        help="Attempt automatic server restart (experimental)",
    )

    args = parser.parse_args()

    # Parse generation lengths
    try:
        generation_lengths = [int(x.strip()) for x in args.generation_lengths.split(",")]
    except ValueError:
        print("Error: Invalid generation_lengths format. Use comma-separated integers.")
        return 1

    # Validate parameters
    if len(generation_lengths) == 0:
        print("Error: No generation lengths specified")
        return 1

    if args.requests_per_length < 1:
        print("Error: requests_per_length must be at least 1")
        return 1

    # Build configuration
    config = {
        "server_url": args.server_url,
        "server_port": args.server_port,
        "generation_lengths": generation_lengths,
        "requests_per_length": args.requests_per_length,
        "output_dir": args.output_dir,
        "generate_plots": args.generate_plots,
        "generate_report": args.generate_report,
        "timeout_seconds": args.timeout_seconds,
        "auto_server_restart": args.auto_server_restart,
    }

    print("🎵 StreamMUSE Generation Length Benchmark")
    print("=" * 50)

    # Run the benchmark
    benchmark = GenerationLengthBenchmark(config)
    success = benchmark.run_parameter_sweep()

    if success:
        print("\n🎉 Benchmark completed successfully!")
        print(f"📁 Results saved to: {Path(args.output_dir).absolute()}")
        return 0
    else:
        print("\n❌ Benchmark failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

