#!/usr/bin/env python3
"""
Multiple Runs Evolution Script

This script runs multiple independent evolutionary runs with different random
seeds and saves each run to a separate database file. This is useful for:
- Assessing consistency and reliability of the EA
- Computing statistics across independent runs
- Analyzing variance in evolutionary outcomes

Each run produces:
- A separate database file (e.g., run1.db, run2.db, ...)
- A corresponding config JSON file (e.g., run1_config.json, ...)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from ea_settings import EASettings
from fitness_functions import FITNESS_FUNCTIONS, PREPROCESSING_FUNCTIONS
from morphology_fitness_analysis import MorphologyAnalyzer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.traceback import install

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EAStep
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING

# Global constants
DB_HANDLING_MODES = Literal["delete", "halt"]

# Global functions
install()
console = Console()

# Type Aliases
type Population = list[Individual]
type PopulationFunc = Callable[[Population], Population]


def preprocess(config: EASettings) -> None:
    """Preprocess function to load target robots into morphology analyzer."""
    PREPROCESSING_FUNCTIONS[config.task](config)


# ------------------------ EA STEPS ------------------------ #
def parent_selection(
    population: Population, config: EASettings, ea: EA | None = None
) -> Population:
    random.shuffle(population)
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_i.tags = {"ps": True}
            ind_j.tags = {"ps": False}
        else:
            ind_i.tags = {"ps": False}
            ind_j.tags = {"ps": True}
    return population


def crossover(
    population: Population, config: EASettings, ea: EA | None = None
) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents) - 1, 2):
        parent_i = parents[idx]
        parent_j = parents[idx + 1]
        genotype_i, genotype_j = config.crossover(
            config.genotype.from_json(parent_i.genotype),
            config.genotype.from_json(parent_j.genotype),
            **config.crossover_params,
        )

        child_i = Individual()
        child_i.genotype = genotype_i.to_json()
        child_i.tags = {"mut": True}
        child_i.requires_eval = True

        child_j = Individual()
        child_j.genotype = genotype_j.to_json()
        child_j.tags = {"mut": True}
        child_j.requires_eval = True

        population.extend([child_i, child_j])
    return population


def mutation(
    population: Population, config: EASettings, ea: EA | None = None
) -> Population:
    for ind in population:
        if ind.tags.get("mut", False):
            genes = config.genotype.from_json(ind.genotype)
            mutated = config.mutation(
                individual=genes,
                **config.mutation_params,
            )
            ind.genotype = mutated.to_json()
            ind.requires_eval = True
    return population


def evaluate(
    population: Population, config: EASettings, ea: EA | None = None
) -> Population:
    fitness_function = FITNESS_FUNCTIONS[config.task]
    population = fitness_function(population, config, ea)
    for ind in population:
        ind.requires_eval = False
    return population


def survivor_selection(
    population: Population, config: EASettings, ea: EA | None = None
) -> Population:
    random.shuffle(population)
    current_pop_size = len(population)
    for idx in range(len(population)):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_j.alive = False
        else:
            ind_i.alive = False

        current_pop_size -= 1
        if current_pop_size <= config.target_population_size:
            break
    return population


# ------------------------ END EA STEPS ------------------------ #


def read_config_file(
    config_path: Path,
    run_idx: int,
    seed: int,
    output_folder: Path,
) -> EASettings:
    """Read config file and prepare settings for a specific run.

    Args:
        config_path: Path to the TOML config file
        run_idx: Index of the current run (for naming)
        seed: Random seed for this run
        output_folder: Output folder for database files

    Returns:
        EASettings configured for this specific run
    """
    cfg = tomllib.loads(config_path.read_text())

    # Resolve the active operators from the chosen genotype profile
    gname = cfg["run"]["genotype"]
    gblock = cfg["genotypes"][gname]
    mutation_name = cfg["run"].get("mutation", gblock["defaults"]["mutation"])
    crossover_name = cfg["run"].get(
        "crossover", gblock["defaults"]["crossover"]
    )
    task = cfg["run"]["task"]
    mutation_params = gblock.get("mutation", {}).get("params", {})
    crossover_params = gblock.get("crossover", {}).get("params", {})

    task_params = cfg["task"].get(task, {}).get("params", {})
    morphology_analyzer = MorphologyAnalyzer(
        metric=task_params.get("distance_metric", "descriptor"),
    )

    genotype = GENOTYPES_MAPPING[gname]

    mutation = genotype.get_mutator_object()
    mutation.set_which_mutation(mutation_name)
    crossover = genotype.get_crossover_object()
    crossover.set_which_crossover(crossover_name)

    # Create unique database file name for this run
    db_file_name = f"run{run_idx}.db"
    db_file_path = output_folder / db_file_name

    settings = EASettings(
        quiet=cfg["ec"]["quiet"],
        is_maximisation=cfg["ec"]["is_maximisation"],
        first_generation_id=cfg["ec"]["first_generation_id"],
        num_of_generations=cfg["ec"]["num_of_generations"],
        target_population_size=cfg["ec"]["target_population_size"],
        genotype=genotype,
        mutation=mutation,
        mutation_params=mutation_params,
        crossover=crossover,
        crossover_params=crossover_params,
        task=task,
        task_params=task_params,
        morphology_analyzer=morphology_analyzer,
        output_folder=output_folder,
        db_file_name=db_file_name,
        db_handling=cfg["data"]["db_handling"],
        db_file_path=db_file_path,
    )

    # Save config to JSON for the dashboard to load later
    save_config_to_json(settings, run_idx, output_folder, gname, mutation_name, crossover_name, seed)

    return settings


def save_config_to_json(
    config: EASettings,
    run_idx: int,
    output_folder: Path,
    genotype_name: str,
    mutation_name: str,
    crossover_name: str,
    seed: int,
) -> None:
    """Save configuration to JSON file for dashboard loading.

    Args:
        config: EASettings object
        run_idx: Index of the current run
        output_folder: Output folder for config file
        genotype_name: Name of the genotype
        mutation_name: Name of the mutation operator
        crossover_name: Name of the crossover operator
        seed: Random seed used for this run
    """
    config_data = {
        "run_id": run_idx,
        "seed": seed,
        "resolved_settings": {
            "is_maximisation": config.is_maximisation,
            "first_generation_id": config.first_generation_id,
            "num_of_generations": config.num_of_generations,
            "target_population_size": config.target_population_size,
            "genotype_name": genotype_name,
            "mutation_name": mutation_name,
            "crossover_name": crossover_name,
            "task": config.task,
            "task_params": config.task_params,
        },
    }

    config_filename = f"run{run_idx}_config.json"
    config_path = output_folder / config_filename

    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)


def create_individual(config: EASettings) -> Individual:
    """Create a new individual."""
    ind = Individual()
    ind.genotype = config.genotype.create_individual().to_json()
    return ind


def run_single_evolution(
    run_idx: int,
    config_path: Path,
    seed: int,
    output_folder: Path,
    quiet: bool = False,
) -> dict:
    """Run a single evolutionary run.

    Args:
        run_idx: Index of this run (for naming)
        config_path: Path to the TOML config file
        seed: Random seed for this run
        output_folder: Output folder for database
        quiet: Suppress output if True

    Returns:
        Dictionary with run statistics
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Read config and prepare for this run
    config = read_config_file(config_path, run_idx, seed, output_folder)
    preprocess(config)

    if not quiet:
        console.log(f"[Run {run_idx}] Starting with seed {seed}")

    # Create initial population
    population_list = [
        create_individual(config) for _ in range(config.target_population_size)
    ]
    for ind in population_list:
        ind.fitness = -float("inf")

    # Create EA steps
    ops = [
        EAStep("parent_selection", partial(parent_selection, config=config)),
        EAStep("crossover", partial(crossover, config=config)),
        EAStep("mutation", partial(mutation, config=config)),
        EAStep("evaluation", partial(evaluate, config=config)),
        EAStep("survivor_selection", partial(survivor_selection, config=config)),
    ]

    # Initialize and run EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=config.num_of_generations,
        db_file_path=config.db_file_path,
    )

    ea.run()

    # Get final statistics
    best = ea.get_solution(only_alive=False)
    median = ea.get_solution("median", only_alive=False)
    worst = ea.get_solution("worst", only_alive=False)

    # Collect final generation statistics
    fitnesses = []
    for i in range(config.num_of_generations):
        ea.fetch_population(
            only_alive=False,
            best_comes=None,
            custom_logic=[Individual.time_of_birth == i],
        )
        gen_fitness_values = [ind.fitness for ind in ea.population]
        if gen_fitness_values:
            fitnesses.append(np.mean(gen_fitness_values))

    stats = {
        "run_idx": run_idx,
        "seed": seed,
        "best_fitness": best.fitness if best else None,
        "median_fitness": median.fitness if median else None,
        "worst_fitness": worst.fitness if worst else None,
        "final_avg_fitness": fitnesses[-1] if fitnesses else None,
        "db_file": str(config.db_file_path),
    }

    if not quiet:
        console.log(
            f"[Run {run_idx}] Completed - Best: {stats['best_fitness']:.4f}, "
            f"Avg: {stats['final_avg_fitness']:.4f}"
        )

    return stats


def run_multiple_evolutions(
    num_runs: int,
    config_path: Path,
    output_folder: Path,
    base_seed: int = 42,
    parallel: bool = False,
    max_workers: int = None,
) -> list[dict]:
    """Run multiple independent evolutionary runs.

    Args:
        num_runs: Number of independent runs to execute
        config_path: Path to the TOML config file
        output_folder: Output folder for database files
        base_seed: Base seed for generating run-specific seeds
        parallel: Run in parallel if True
        max_workers: Maximum number of parallel workers (None = num CPUs)

    Returns:
        List of statistics dictionaries, one per run
    """
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Generate seeds for each run
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 1000000, size=num_runs).tolist()

    console.log(f"Starting {num_runs} evolutionary runs...")
    console.log(f"Output folder: {output_folder}")
    console.log(f"Base seed: {base_seed}")
    console.log(f"Parallel execution: {parallel}")

    all_stats = []

    if parallel:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all runs
            futures = {
                executor.submit(
                    run_single_evolution,
                    run_idx,
                    config_path,
                    seeds[run_idx],
                    output_folder,
                    quiet=True,
                ): run_idx
                for run_idx in range(num_runs)
            }

            # Track progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Running {num_runs} evolutions...", total=num_runs
                )

                for future in as_completed(futures):
                    stats = future.result()
                    all_stats.append(stats)
                    progress.advance(task)
                    console.log(
                        f"[Run {stats['run_idx']}] Completed - "
                        f"Best: {stats['best_fitness']:.4f}"
                    )

    else:
        # Sequential execution
        for run_idx in range(num_runs):
            stats = run_single_evolution(
                run_idx,
                config_path,
                seeds[run_idx],
                output_folder,
                quiet=False,
            )
            all_stats.append(stats)

    # Sort by run_idx to maintain order
    all_stats.sort(key=lambda x: x["run_idx"])

    return all_stats


def print_summary(stats_list: list[dict]) -> None:
    """Print summary statistics across all runs."""
    console.print("\n[bold]Summary Across All Runs[/bold]")
    console.print("=" * 60)

    best_fitnesses = [s["best_fitness"] for s in stats_list if s["best_fitness"]]
    avg_fitnesses = [s["final_avg_fitness"] for s in stats_list if s["final_avg_fitness"]]

    if best_fitnesses:
        console.print(f"Best Fitness - Mean: {np.mean(best_fitnesses):.4f} ± {np.std(best_fitnesses):.4f}")
        console.print(f"Best Fitness - Min: {np.min(best_fitnesses):.4f}, Max: {np.max(best_fitnesses):.4f}")

    if avg_fitnesses:
        console.print(f"Final Avg Fitness - Mean: {np.mean(avg_fitnesses):.4f} ± {np.std(avg_fitnesses):.4f}")

    console.print("\n[bold]Database Files:[/bold]")
    for stats in stats_list:
        console.print(f"  Run {stats['run_idx']}: {stats['db_file']}")

    console.print("\n[bold green]All runs completed successfully![/bold green]")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print(f"  Visualize results with:")
    console.print(f"    uv run examples/multiple_runs_dashboard.py \\")
    db_paths = " ".join([f"{stats['db_file']}" for stats in stats_list[:3]])
    console.print(f"      --db_paths {db_paths}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multiple independent evolutionary runs"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of independent runs (default: 5)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/config.toml"),
        help="Path to config file (default: examples/config.toml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("__data__/multiple_runs"),
        help="Output folder for database files (default: __data__/multiple_runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evolutions in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPUs)",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not args.config.exists():
        console.log(f"[red]Error: Config file not found: {args.config}[/red]")
        return

    # Run multiple evolutions
    stats_list = run_multiple_evolutions(
        num_runs=args.num_runs,
        config_path=args.config,
        output_folder=args.output,
        base_seed=args.seed,
        parallel=args.parallel,
        max_workers=args.workers,
    )

    # Print summary
    print_summary(stats_list)


if __name__ == "__main__":
    main()
