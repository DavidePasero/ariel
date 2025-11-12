#!/usr/bin/env python3
"""
Headless Evolution Runner

Runs evolutionary computation without GUI components, storing all data
to SQLite database for later analysis with dashboard.
"""

from __future__ import annotations
import random
from collections.abc import Callable
from pathlib import Path
import tomllib
import json
from typing import Literal
from functools import partial
from datetime import datetime

from fitness_functions import FITNESS_FUNCTIONS
import numpy as np
from rich.console import Console
from rich.traceback import install

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EAStep, EA
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
from ea_settings import EASettings

# Global constants
SEED = 42
DB_HANDLING_MODES = Literal["delete", "halt"]

# Global functions
install()
console = Console()
RNG = np.random.default_rng(SEED)

# Type Aliases
type Population = list[Individual]
type PopulationFunc = Callable[[Population], Population]


def parent_selection(population: Population, config: EASettings) -> Population:
    random.shuffle(population)
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Compare fitness values
        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_i.tags = {"ps": True}
            ind_j.tags = {"ps": False}
        else:
            ind_i.tags = {"ps": False}
            ind_j.tags = {"ps": True}
    return population


def crossover(population: Population, config: EASettings) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents)-1, 2):
        parent_i = parents[idx]
        parent_j = parents[idx + 1]
        genotype_i, genotype_j = config.crossover(
            config.genotype.from_json(parent_i.genotype),
            config.genotype.from_json(parent_j.genotype),
            **config.crossover_params,
        )

        # First child
        child_i = Individual()
        child_i.genotype = genotype_i.to_json()
        child_i.tags = {"mut": True}
        child_i.requires_eval = True

        # Second child
        child_j = Individual()
        child_j.genotype = genotype_j.to_json()
        child_j.tags = {"mut": True}
        child_j.requires_eval = True

        population.extend([child_i, child_j])
    return population


def mutation(population: Population, config: EASettings) -> Population:
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


def evaluate(population: Population, config: EASettings) -> Population:
    fitness_function = FITNESS_FUNCTIONS[config.task]
    population = fitness_function(population, config)
    for ind in population:
        ind.requires_eval = False
    return population


def survivor_selection(population: Population, config: EASettings) -> Population:
    random.shuffle(population)
    current_pop_size = len(population)
    for idx in range(len(population)):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Kill worse individual
        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_j.alive = False
        else:
            ind_i.alive = False

        # Termination condition
        current_pop_size -= 1
        if current_pop_size <= config.target_population_size:
            break
    return population


def create_individual(config: EASettings) -> Individual:
    ind = Individual()
    ind.genotype = config.genotype.create_individual(**config.create_individual_params).to_json()
    return ind


def read_config_file() -> tuple[EASettings, dict]:
    """Read config file and return both EASettings and raw config dict."""
    config_path = Path("examples/config.toml")
    cfg = tomllib.loads(config_path.read_text())

    # Resolve the active operators from the chosen genotype profile
    gname = cfg["run"]["genotype"]
    gblock = cfg["genotypes"][gname]
    mutation_name = cfg["run"].get("mutation", gblock["defaults"]["mutation"])
    crossover_name = cfg["run"].get("crossover", gblock["defaults"]["crossover"])
    task = cfg["run"]["task"]
    create_individual_params = gblock.get("create_individual_params", {})
    mutation_params = gblock.get("mutation", {}).get("params", {})
    crossover_params = gblock.get("crossover", {}).get("params", {})

    task_params = cfg["task"].get(task, {}).get("params", {})

    genotype = GENOTYPES_MAPPING[gname]

    mutation = genotype.get_mutator_object()
    mutation.set_which_mutation(mutation_name)
    crossover = genotype.get_crossover_object()
    crossover.set_which_crossover(crossover_name)

    settings = EASettings(
        quiet=cfg["ec"]["quiet"],
        is_maximisation=cfg["ec"]["is_maximisation"],
        first_generation_id=cfg["ec"]["first_generation_id"],
        num_of_generations=cfg["ec"]["num_of_generations"],
        target_population_size=cfg["ec"]["target_population_size"],
        genotype=genotype,
        create_individual_params=create_individual_params,
        mutation=mutation,
        mutation_params=mutation_params,
        crossover=crossover,
        crossover_params=crossover_params,
        task=task,
        task_params=task_params,
        output_folder=Path(cfg["data"]["output_folder"]),
        db_file_name=cfg["data"]["db_file_name"],
        db_handling=cfg["data"]["db_handling"],
        db_file_path=Path(cfg["data"]["output_folder"]) / cfg["data"]["db_file_name"],
    )
    return settings, cfg


def save_run_config(config: EASettings, raw_config: dict) -> Path:
    """Save the configuration used for this run alongside the database.

    Parameters:
    -----------
    config : EASettings
        The processed EA settings
    raw_config : dict
        The raw configuration dictionary from the TOML file

    Returns:
    --------
    Path
        Path to the saved configuration file
    """
    # Create config filename based on database filename
    db_path = config.db_file_path
    config_filename = db_path.stem + "_config.json"
    config_path = db_path.parent / config_filename

    # Ensure output directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create comprehensive config data to save
    config_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config_file_used": "examples/config.toml",
            "database_file": str(db_path),
            "seed": SEED,
        },
        "raw_config": raw_config,
        "resolved_settings": {
            "genotype_name": raw_config["run"]["genotype"],
            "mutation_name": config.mutation.__class__.__name__ if hasattr(config.mutation, '__class__') else str(config.mutation),
            "crossover_name": config.crossover.__class__.__name__ if hasattr(config.crossover, '__class__') else str(config.crossover),
            "task": config.task,
            "num_of_generations": config.num_of_generations,
            "target_population_size": config.target_population_size,
            "is_maximisation": config.is_maximisation,
            "first_generation_id": config.first_generation_id,
            "quiet": config.quiet,
            "create_individual_params": config.create_individual_params,
            "mutation_params": config.mutation_params,
            "crossover_params": config.crossover_params,
            "task_params": config.task_params,
            "output_folder": str(config.output_folder),
            "db_file_name": config.db_file_name,
            "db_handling": config.db_handling,
        }
    }

    # Save to JSON file
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    return config_path


def load_run_config(db_path: Path) -> dict:
    """Load the configuration that was used for a specific run.

    Parameters:
    -----------
    db_path : Path
        Path to the database file

    Returns:
    --------
    dict
        The configuration data that was saved during the run

    Raises:
    -------
    FileNotFoundError
        If the configuration file doesn't exist
    """
    config_filename = db_path.stem + "_config.json"
    config_path = db_path.parent / config_filename

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def main() -> None:
    """Entry point for headless evolution."""
    # Read configuration and save it for later use
    config, raw_config = read_config_file()

    console.log(f"Starting headless evolution with {config.num_of_generations} generations")
    console.log(f"Database will be stored at: {config.db_file_path}")

    # Save configuration alongside database
    config_path = save_run_config(config, raw_config)
    console.log(f"Configuration saved to: {config_path}")

    # Create initial population
    population_list = [create_individual(config) for _ in range(config.target_population_size)]
    population_list = evaluate(population_list, config)

    # Create EA steps
    ops = [
        EAStep("parent_selection", partial(parent_selection, config=config)),
        EAStep("crossover",        partial(crossover,        config=config)),
        EAStep("mutation",         partial(mutation,         config=config)),
        EAStep("evaluation",       partial(evaluate,         config=config)),
        EAStep("survivor_selection", partial(survivor_selection, config=config)),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=config.num_of_generations,
        db_file_path=config.db_file_path,
        db_handling=config.db_handling,
    )

    # Run evolution
    ea.run()

    # Report final statistics
    best = ea.get_solution(only_alive=False)
    console.log(f"Evolution complete! Best fitness: {best.fitness}")

    median = ea.get_solution("median", only_alive=False)
    console.log(f"Median fitness: {median.fitness}")

    worst = ea.get_solution("worst", only_alive=False)
    console.log(f"Worst fitness: {worst.fitness}")

    console.log(f"All data saved to database: {config.db_file_path}")
    console.log(f"Configuration saved to: {config_path}")
    console.log("Use 'uv run examples/load_dashboard.py' to visualize results")


if __name__ == "__main__":
    main()
