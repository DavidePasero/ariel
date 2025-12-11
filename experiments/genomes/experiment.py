"""GenomeEAExperiment - Unified experiment runner for evolutionary algorithms.

This module provides a unified interface for running single or multiple
evolutionary runs with genome-based genotypes. It merges the functionality
of single runs and multiple runs into a single class.
"""

from __future__ import annotations

import json
import random
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.traceback import install

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EAStep
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING

from experiments.genomes.ea_settings import EASettings
from experiments.genomes.fitness_functions import (
    FITNESS_FUNCTIONS,
    PREPROCESSING_FUNCTIONS,
)
from experiments.genomes.morphology_fitness_analysis import MorphologyAnalyzer

# Global setup
install()
console = Console()

# Type Aliases
type Population = list[Individual]
type PopulationFunc = Callable[[Population], Population]


class GenomeEAExperiment:
    """Unified experiment runner for evolutionary algorithms.

    This class provides a single interface for running evolutionary
    experiments with genome-based genotypes, supporting both single
    runs and multiple independent runs with different random seeds.

    Parameters
    ----------
    config_path : Path
        Path to the TOML configuration file
    output_folder : Path | None, optional
        Output folder for database files. If None, uses config default
    base_seed : int, optional
        Base random seed for reproducibility, by default 42

    Examples
    --------
    Single run:
    >>> experiment = GenomeEAExperiment(Path("config.toml"))
    >>> stats = experiment.run_single()

    Multiple runs:
    >>> experiment = GenomeEAExperiment(Path("config.toml"))
    >>> all_stats = experiment.run_multiple(num_runs=5, parallel=True)
    """

    def __init__(
        self,
        config_path: Path,
        output_folder: Path | None = None,
        base_seed: int = 42,
    ):
        """Initialize the experiment runner.

        Parameters
        ----------
        config_path : Path
            Path to the TOML configuration file
        output_folder : Path | None, optional
            Output folder for database files. If None, uses config default
        base_seed : int, optional
            Base random seed for reproducibility, by default 42
        """
        self.config_path = config_path
        self.output_folder = output_folder
        self.base_seed = base_seed

        # Validate config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )

    def _read_config_file(
        self,
        run_idx: int | None = None,
        seed: int | None = None,
    ) -> EASettings:
        """Read and parse the configuration file.

        Parameters
        ----------
        run_idx : int | None, optional
            Run index for naming database files. If None, uses default name
        seed : int | None, optional
            Random seed for this run. If None, uses base_seed

        Returns
        -------
        EASettings
            Configured settings for the evolutionary algorithm
        """
        cfg = tomllib.loads(self.config_path.read_text())

        # Resolve the active operators from the chosen genotype profile
        gname = cfg["run"]["genotype"]
        gblock = cfg["genotypes"][gname]
        mutation_name = cfg["run"].get(
            "mutation", gblock["defaults"]["mutation"]
        )
        crossover_name = cfg["run"].get(
            "crossover", gblock["defaults"]["crossover"]
        )
        task = cfg["run"]["task"]
        include_diversity = cfg["run"]["include_diversity_measure"]
        mutation_params = gblock.get("mutation", {}).get("params", {})
        crossover_params = gblock.get("crossover", {}).get("params", {})

        task_params = cfg["task"].get(task, {}).get("params", {})
        include_diversity_measure_params = (
            cfg["task"].get("evolve_for_novelty", {}).get("params", {})
            if include_diversity
            else {}
        )
        morphology_analyzer = MorphologyAnalyzer(
            metric=task_params.get("distance_metric", "descriptor"),
        )

        genotype = GENOTYPES_MAPPING[gname]

        mutation = genotype.get_mutator_object()
        mutation.set_which_mutation(mutation_name)
        crossover = genotype.get_crossover_object()
        crossover.set_which_crossover(crossover_name)

        # Determine output folder and database file name
        output_folder = self.output_folder or Path(cfg["data"]["output_folder"])
        if run_idx is not None:
            db_file_name = f"run{run_idx}.db"
        else:
            db_file_name = cfg["data"]["db_file_name"]

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
            include_diversity=include_diversity,
            include_diversity_measure_params=include_diversity_measure_params,
            task_params=task_params,
            morphology_analyzer=morphology_analyzer,
            output_folder=output_folder,
            db_file_name=db_file_name,
            db_handling=cfg["data"]["db_handling"],
            db_file_path=db_file_path,
        )

        self._save_config_to_json(
            settings,
            run_idx,
            output_folder,
            gname,
            mutation_name,
            crossover_name,
            seed,
        )

        return settings

    def _save_config_to_json(
        self,
        config: EASettings,
        run_idx: int,
        output_folder: Path,
        genotype_name: str,
        mutation_name: str,
        crossover_name: str,
        seed: int,
    ) -> None:
        """Save configuration to JSON file for later analysis.

        Parameters
        ----------
        config : EASettings
            Configuration settings
        run_idx : int
            Run index
        output_folder : Path
            Output folder
        genotype_name : str
            Name of the genotype
        mutation_name : str
            Name of the mutation operator
        crossover_name : str
            Name of the crossover operator
        seed : int
            Random seed used
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

        # if run idx is None we will use he dbname as the config name
        if run_idx is None:
            config_filename = config.db_file_name.strip(".db") + "_config.json"
        else:
            config_filename = f"run{run_idx}_config.json"
        config_path = output_folder / config_filename

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def _preprocess(self, config: EASettings) -> None:
        """Run preprocessing steps (e.g., load target robots).

        Parameters
        ----------
        config : EASettings
            Configuration settings
        """
        PREPROCESSING_FUNCTIONS[config.task](config)

    def _create_individual(self, config: EASettings) -> Individual:
        """Create a new individual with random genotype.

        Parameters
        ----------
        config : EASettings
            Configuration settings

        Returns
        -------
        Individual
            New individual with random genotype
        """
        ind = Individual()
        ind.genotype = config.genotype.create_individual().to_json()
        return ind

    def _create_ea_steps(self, config: EASettings) -> list[EAStep]:
        """Create the EA steps (operations) for evolution.

        Parameters
        ----------
        config : EASettings
            Configuration settings

        Returns
        -------
        list[EAStep]
            List of EA operations to perform each generation
        """
        ops = [
            EAStep(
                "parent_selection",
                partial(self._parent_selection, config=config),
            ),
            EAStep("crossover", partial(self._crossover, config=config)),
            EAStep("mutation", partial(self._mutation, config=config)),
            EAStep("evaluation", partial(self._evaluate, config=config)),
            *(
                [
                    EAStep(
                        "diversity_evaluation",
                        partial(self._diversity_evaluate, config=config),
                    ),
                ]
                if config.include_diversity
                else []
            ),
            EAStep(
                "survivor_selection",
                partial(self._survivor_selection, config=config),
            ),
        ]
        return ops

    # ------------------------ EA STEPS ------------------------ #

    def _parent_selection(
        self, population: Population, config: EASettings, ea: EA | None = None
    ) -> Population:
        """Tournament-based parent selection.

        Parameters
        ----------
        population : Population
            Current population
        config : EASettings
            Configuration settings
        ea : EA | None, optional
            EA instance, by default None

        Returns
        -------
        Population
            Population with parent selection tags
        """
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

    def _crossover(
        self, population: Population, config: EASettings, ea: EA | None = None
    ) -> Population:
        """Apply crossover to selected parents.

        Parameters
        ----------
        population : Population
            Current population
        config : EASettings
            Configuration settings
        ea : EA | None, optional
            EA instance, by default None

        Returns
        -------
        Population
            Population with offspring added
        """
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

    def _mutation(
        self, population: Population, config: EASettings, ea: EA | None = None
    ) -> Population:
        """Apply mutation to marked individuals.

        Parameters
        ----------
        population : Population
            Current population
        config : EASettings
            Configuration settings
        ea : EA | None, optional
            EA instance, by default None

        Returns
        -------
        Population
            Population with mutations applied
        """
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

    def _evaluate(
        self, population: Population, config: EASettings, ea: EA | None = None
    ) -> Population:
        """Evaluate fitness of individuals.

        Parameters
        ----------
        population : Population
            Current population
        config : EASettings
            Configuration settings
        ea : EA | None, optional
            EA instance, by default None

        Returns
        -------
        Population
            Population with fitness values updated
        """
        fitness_function = FITNESS_FUNCTIONS[config.task]
        population = fitness_function(population, config, ea)
        for ind in population:
            ind.requires_eval = False
        return population

    def _diversity_evaluate(
        self, population: Population, config: EASettings, ea: EA | None = None
    ) -> Population:
        """Evaluate diversity of individuals. Only used when we don't have an evolve_for_novelty task.

        Parameters
        ----------
        population : Population
            Current population
        config : EASettings
            Configuration settings
        ea : EA | None, optional
            EA instance, by default None

        Returns
        -------
        Population
            Population with diversity values updated
        """
        fitness_function = FITNESS_FUNCTIONS["evolve_for_novelty"]
        population = fitness_function(
            population,
            config,
            ea,
            is_fitness_function=False,
        )
        return population

    def _survivor_selection(
        self, population: Population, config: EASettings, ea: EA | None = None
    ) -> Population:
        """Tournament-based survivor selection.

        Parameters
        ----------
        population : Population
            Current population
        config : EASettings
            Configuration settings
        ea : EA | None, optional
            EA instance, by default None

        Returns
        -------
        Population
            Population with survivors marked
        """
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

    def run_single(
        self,
        seed: int | None = None,
        quiet: bool = False,
    ) -> dict:
        """Run a single evolutionary run.

        Parameters
        ----------
        seed : int | None, optional
            Random seed. If None, uses base_seed
        quiet : bool, optional
            Suppress output, by default False

        Returns
        -------
        dict
            Statistics from the run including best, median, worst fitness
        """
        # Set random seed
        seed = seed or self.base_seed
        random.seed(seed)
        np.random.seed(seed)

        # Read config and preprocess (save config with run_idx=0 for single runs)
        config = self._read_config_file(run_idx=None, seed=seed)
        self._preprocess(config)

        # Ensure output folder exists
        config.output_folder.mkdir(parents=True, exist_ok=True)

        if not quiet:
            console.log(f"Starting evolution with seed {seed}")
            console.log(f"Output: {config.db_file_path}")

        # Create initial population
        population_list = [
            self._create_individual(config)
            for _ in range(config.target_population_size)
        ]
        for ind in population_list:
            ind.fitness = -float("inf")

        # Create EA steps and run
        ops = self._create_ea_steps(config)
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

        # Collect fitness statistics per generation
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
            "seed": seed,
            "best_fitness": best.fitness if best else None,
            "median_fitness": median.fitness if median else None,
            "worst_fitness": worst.fitness if worst else None,
            "final_avg_fitness": fitnesses[-1] if fitnesses else None,
            "avg_fitness_per_gen": fitnesses,
            "db_file": str(config.db_file_path),
        }

        if not quiet:
            console.log(f"Evolution completed!")
            console.log(f"Best fitness: {stats['best_fitness']:.4f}")
            console.log(f"Final avg fitness: {stats['final_avg_fitness']:.4f}")

        return stats

    def _run_single_for_multirun(
        self,
        run_idx: int,
        seed: int,
        quiet: bool = False,
    ) -> dict:
        """Run a single evolution as part of multiple runs.

        Parameters
        ----------
        run_idx : int
            Index of this run (for naming)
        seed : int
            Random seed for this run
        quiet : bool, optional
            Suppress output, by default False

        Returns
        -------
        dict
            Statistics from the run
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Read config for this specific run
        config = self._read_config_file(run_idx=run_idx, seed=seed)
        self._preprocess(config)

        # Ensure output folder exists
        config.output_folder.mkdir(parents=True, exist_ok=True)

        if not quiet:
            console.log(f"[Run {run_idx}] Starting with seed {seed}")

        # Create initial population
        population_list = [
            self._create_individual(config)
            for _ in range(config.target_population_size)
        ]
        for ind in population_list:
            ind.fitness = -float("inf")

        # Create EA steps and run
        ops = self._create_ea_steps(config)
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

        # Collect fitness statistics
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
                f"[Run {run_idx}] Completed - Best: "
                f"{stats['best_fitness']:.4f}, "
                f"Avg: {stats['final_avg_fitness']:.4f}"
            )

        return stats

    def run_multiple(
        self,
        num_runs: int = 5,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> list[dict]:
        """Run multiple independent evolutionary runs.

        Parameters
        ----------
        num_runs : int, optional
            Number of independent runs, by default 5
        parallel : bool, optional
            Run in parallel, by default False
        max_workers : int | None, optional
            Maximum number of parallel workers (None = num CPUs),
            by default None

        Returns
        -------
        list[dict]
            List of statistics dictionaries, one per run
        """
        # Generate seeds for each run
        rng = np.random.default_rng(self.base_seed)
        seeds = rng.integers(0, 1000000, size=num_runs).tolist()

        # Ensure output folder exists
        output_folder = self.output_folder or Path("__data__/multiple_runs")
        output_folder.mkdir(parents=True, exist_ok=True)

        console.log(f"Starting {num_runs} evolutionary runs...")
        console.log(f"Output folder: {output_folder}")
        console.log(f"Base seed: {self.base_seed}")
        console.log(f"Parallel execution: {parallel}")

        all_stats = []

        if parallel:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all runs
                futures = {
                    executor.submit(
                        self._run_single_for_multirun,
                        run_idx,
                        seeds[run_idx],
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
                stats = self._run_single_for_multirun(
                    run_idx,
                    seeds[run_idx],
                    quiet=False,
                )
                all_stats.append(stats)

        # Sort by run_idx to maintain order
        all_stats.sort(key=lambda x: x["run_idx"])

        # Print summary
        self._print_summary(all_stats)

        return all_stats

    def _print_summary(self, stats_list: list[dict]) -> None:
        """Print summary statistics across all runs.

        Parameters
        ----------
        stats_list : list[dict]
            List of statistics from all runs
        """
        console.print("\n[bold]Summary Across All Runs[/bold]")
        console.print("=" * 60)

        best_fitnesses = [
            s["best_fitness"] for s in stats_list if s["best_fitness"]
        ]
        avg_fitnesses = [
            s["final_avg_fitness"] for s in stats_list if s["final_avg_fitness"]
        ]

        if best_fitnesses:
            console.print(
                f"Best Fitness - Mean: {np.mean(best_fitnesses):.4f} "
                f"± {np.std(best_fitnesses):.4f}"
            )
            console.print(
                f"Best Fitness - Min: {np.min(best_fitnesses):.4f}, "
                f"Max: {np.max(best_fitnesses):.4f}"
            )

        if avg_fitnesses:
            console.print(
                f"Final Avg Fitness - Mean: {np.mean(avg_fitnesses):.4f} "
                f"± {np.std(avg_fitnesses):.4f}"
            )

        console.print("\n[bold]Database Files:[/bold]")
        for stats in stats_list:
            console.print(f"  Run {stats['run_idx']}: {stats['db_file']}")

        console.print(
            "\n[bold green]All runs completed successfully![/bold green]"
        )
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  Visualize results with:")
        console.print("    uv run examples/multiple_runs_dashboard.py \\")
        db_paths = " ".join([f"{stats['db_file']}" for stats in stats_list[:3]])
        console.print(f"      --db_paths {db_paths}")
