#!/usr/bin/env python3
"""Example script demonstrating how to use GenomeEAExperiment.

This script shows how to run both single and multiple evolutionary runs
using the unified GenomeEAExperiment class.
"""

import argparse
from pathlib import Path

from experiments.genomes.experiment import GenomeEAExperiment


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evolutionary experiments with genome-based genotypes"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/genomes/config.toml"),
        help="Path to config file (default: examples/config.toml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for database files (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multiple"],
        default="single",
        help="Run mode: single or multiple (default: single)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for multiple mode (default: 5)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run multiple evolutions in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPUs)",
    )

    args = parser.parse_args()

    # Create experiment instance
    experiment = GenomeEAExperiment(
        config_path=args.config,
        output_folder=args.output,
        base_seed=args.seed,
    )

    # Run experiment based on mode
    if args.mode == "single":
        print(f"Running single experiment with config: {args.config}")
        stats = experiment.run_single()
        print("\nResults:")
        print(f"  Best fitness: {stats['best_fitness']:.4f}")
        print(f"  Median fitness: {stats['median_fitness']:.4f}")
        print(f"  Worst fitness: {stats['worst_fitness']:.4f}")
        print(f"  Final avg fitness: {stats['final_avg_fitness']:.4f}")
        print(f"  Database: {stats['db_file']}")

    elif args.mode == "multiple":
        print(f"Running {args.num_runs} experiments with config: {args.config}")
        all_stats = experiment.run_multiple(
            num_runs=args.num_runs,
            parallel=args.parallel,
            max_workers=args.workers,
        )
        print(f"\nCompleted {len(all_stats)} runs successfully!")


if __name__ == "__main__":
    main()
