import concurrent.futures
from pathlib import Path
from typing import Any, List, Protocol

from experiments.genomes.ea_settings import EASettings
from experiments.genomes.metrics import compute_6d_descriptor
from scipy.spatial.distance import cdist

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA

type Population = list[Individual]

# --- Helper Functions for Parallel Execution ---
# These must be defined at the module level to be picklable.


def _eval_single_copy(
    genotype_json: str, target_graph: Any, config: EASettings
) -> float:
    """Helper to evaluate a single individual against a target graph."""
    # Note: We might need to re-import config/classes if they aren't available in the worker process,
    # but usually passing 'config' is fine if it's picklable.
    genotype = config.genotype.from_json(genotype_json)
    ind_digraph = genotype.to_digraph()
    fitness = config.morphology_analyzer.compute_fitness_score(
        ind_digraph,
        target_graph,
    )
    return fitness


# -----------------------------------------------


class PreprocessFunc(Protocol):
    def __call__(
        self,
        config: EASettings,
    ) -> None: ...


class FitnessFunc(Protocol):
    def __call__(
        self,
        population: Population,
        config: EASettings,
    ) -> Population: ...


PREPROCESSING_FUNCTIONS: dict[str, PreprocessFunc] = {}
FITNESS_FUNCTIONS: dict[str, FitnessFunc] = {}


def register_fitness(name: str | None = None):
    def deco(fn: FitnessFunc) -> FitnessFunc:
        FITNESS_FUNCTIONS[name or fn.__name__] = fn
        return fn

    return deco


def register_preprocessing(name: str | None = None):
    def deco(fn: PreprocessFunc) -> PreprocessFunc:
        PREPROCESSING_FUNCTIONS[name or fn.__name__] = fn
        return fn

    return deco


# ------------------------ Preprocessing Functions ------------------------ #


@register_preprocessing("evolve_to_copy")
def preprocess_evolve_to_copy(
    config: EASettings,
) -> None:
    config.morphology_analyzer.load_target_robots(
        Path(config.task_params["target_robot_path"])
    )


@register_preprocessing("evolve_to_copy_sequence_normal")
def preprocess_evolve_to_copy_sequence_normal(
    config: EASettings,
) -> None:
    target_paths = config.task_params["target_robot_paths"]
    config.morphology_analyzer.load_target_robots(
        *(Path(p) for p in target_paths)
    )


@register_preprocessing("evolve_to_copy_sequence_reverse")
def preprocess_evolve_to_copy_sequence_reverse(
    config: EASettings,
) -> None:
    target_paths = config.task_params["target_robot_paths"]
    config.morphology_analyzer.load_target_robots(
        *(Path(p) for p in target_paths)
    )


@register_preprocessing("evolve_for_novelty")
def preprocess_evolve_for_novelty(
    config: EASettings,
) -> None:
    pass


# ------------------------ Fitness Functions ------------------------ #


def _parallel_eval_copy(
    population: Population, config: EASettings, target_graph: Any
) -> Population:
    """Shared logic for parallelizing copy-based fitness."""

    # Prepare arguments for map
    # We create a list of arguments for each individual
    # Note: 'target_graph' and 'config' might be large.
    # If pickling overhead is high, we might need a different strategy (like initializing workers with global state).

    genotypes = [ind.genotype for ind in population]

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [
            executor.submit(_eval_single_copy, g_json, target_graph, config)
            for g_json in genotypes
        ]

        # Collect results as they complete (in order)
        results = [f.result() for f in futures]

    # Assign results back to individuals
    for ind, fit in zip(population, results):
        ind.fitness = fit

    return population


@register_fitness("evolve_to_copy")
def evolve_to_copy(
    population: Population, config: EASettings, ea: EA
) -> Population:
    target_graph = config.morphology_analyzer.target_graphs[0]
    return _parallel_eval_copy(population, config, target_graph)


@register_fitness("evolve_to_copy_sequence_normal")
def evolve_to_copy_sequence_normal(
    population: Population, config: EASettings, ea: EA
) -> Population:
    total_targets = len(config.task_params["target_robot_paths"])
    interval = config.num_of_generations // total_targets
    raw_index = ea.current_generation // interval
    current_objective = min(raw_index, total_targets - 1)

    target_graph = config.morphology_analyzer.target_graphs[current_objective]
    return _parallel_eval_copy(population, config, target_graph)


@register_fitness("evolve_to_copy_sequence_reverse")
def evolve_to_copy_sequence_reverse(
    population: Population, config: EASettings, ea: EA
) -> Population:
    total_targets = len(config.task_params["target_robot_paths"])
    interval = config.num_of_generations // total_targets
    current_objective = max(
        0, total_targets - 1 - (ea.current_generation // interval)
    )

    target_graph = config.morphology_analyzer.target_graphs[current_objective]
    return _parallel_eval_copy(population, config, target_graph)


@register_fitness("evolve_for_novelty")
def evolve_for_novelty(
    population: Population, config: EASettings, ea: EA
) -> Population:
    robot_descriptors = []

    def _get_descriptors_and_append(
        population: Population, robot_descriptors: list
    ):
        for ind in population:
            genotype = config.genotype.from_json(ind.genotype)
            measures = compute_6d_descriptor(genotype.to_digraph())
            robot_descriptors.append(measures)
        return robot_descriptors

    # Get archived population
    ea.fetch_population(
        only_alive=False,
        custom_logic=(
            not Individual.alive,
            Individual.tags_["novel"],
        ),
    )

    old_population = ea.population
    # Append archived individuals' descriptors
    robot_descriptors = _get_descriptors_and_append(
        old_population,
        robot_descriptors,
    )

    # Append robot descriptors with current population
    robot_descriptors = _get_descriptors_and_append(
        population,
        robot_descriptors,
    )

    distance_matrix = cdist(
        robot_descriptors,
        robot_descriptors,
        metric="euclidean",
    )

    # Compute novelty scores
    for idx, ind in enumerate(population):
        # Exclude self-distance by setting it to infinity
        distances = distance_matrix[idx]
        distances[idx] = float("inf")
        # Compute average distance to k nearest neighbors, making sure k does not exceed population size
        k = min(config.task_params["n_points_to_consider"], len(population) - 1)
        nearest_distances = sorted(distances)[:k]
        novelty_score = sum(nearest_distances) / k
        ind.fitness = novelty_score
        if novelty_score >= config.task_params["threshold_for_novelty"]:
            ind.tags["novel"] = True
    return population
