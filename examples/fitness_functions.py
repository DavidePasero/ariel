from pathlib import Path
from typing import Protocol

from ea_settings import EASettings
from scipy.spatial.distance import cdist

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA

type Population = list[Individual]


class PreprocessFunc(Protocol):
    def __call__(
        self,
        config: EASettings,
        ea: EA,
    ) -> None: ...


# Types
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


# ------------------------ Fitness Functions ------------------------ #


@register_fitness("evolve_to_copy")
def evolve_to_copy(
    population: Population, config: EASettings, ea: EA
) -> Population:
    for ind in population:
        genotype = config.genotype.from_json(ind.genotype)
        # Convert to digraph
        ind_digraph = genotype.to_digraph()
        fitness = config.morphology_analyzer.compute_fitness_score(
            ind_digraph,
            config.morphology_analyzer.target_graphs[0],
        )
        ind.fitness = fitness
    return population


@register_fitness("evolve_to_copy_sequence_normal")
def evolve_to_copy_sequence_normal(
    population: Population, config: EASettings, ea: EA
) -> Population:
    total_targets = len(config.task_params["target_robot_paths"])
    # 1. Calculate the interval size
    interval = config.num_of_generations // total_targets
    # 2. Calculate the raw index
    raw_index = ea.current_generation // interval
    # 3. Clamp the index so it doesn't exceed the last target
    #    (this handles the remainder of the division)
    current_objective = min(raw_index, total_targets - 1)

    for ind in population:
        genotype = config.genotype.from_json(ind.genotype)
        ind_digraph = genotype.to_digraph()
        fitness = config.morphology_analyzer.compute_fitness_score(
            ind_digraph,
            config.morphology_analyzer.target_graphs[current_objective],
        )
        ind.fitness = fitness
    return population


@register_fitness("evolve_to_copy_sequence_reverse")
def evolve_to_copy_sequence_reverse(
    population: Population, config: EASettings, ea: EA
) -> Population:
    total_targets = len(config.task_params["target_robot_paths"])
    interval = config.num_of_generations // total_targets
    # Calculate the reverse index and ensure it doesn't go below 0
    current_objective = max(
        0, total_targets - 1 - (ea.current_generation // interval)
    )

    for ind in population:
        genotype = config.genotype.from_json(ind.genotype)
        # Convert to digraph
        ind_digraph = genotype.to_digraph()
        fitness = config.morphology_analyzer.compute_fitness_score(
            ind_digraph,
            config.morphology_analyzer.target_graphs[current_objective],
        )
        ind.fitness = fitness
    return population


@register_fitness("evolve_for_novelty")
def evolve_for_novelty(
    population: Population, config: EASettings, ea: EA
) -> Population:
    robot_descriptors = []

    def _get_descriptors_and_append(
        population: Population, robot_descriptors: list
    ):
        for ind in population:
            measures = _get_6d_descriptors(ind, config)
            robot_descriptors.append(measures)
        return robot_descriptors

    ea.fetch_population(
        only_alive=False, custom_logic=(Individual.alive == False)
    )

    old_population = ea.population

    robot_descriptors = _get_descriptors_and_append(
        population, robot_descriptors
    )
    robot_descriptors = _get_descriptors_and_append(
        old_population, robot_descriptors
    )

    distance_matrix = cdist(
        robot_descriptors, robot_descriptors, metric="euclidean"
    )

    # Current population
    ea.fetch_population()
    population = ea.population

    for idx, ind in enumerate(population):
        # Exclude self-distance by setting it to infinity
        distances = distance_matrix[idx]
        distances[idx] = float("inf")
        # Compute average distance to k nearest neighbors, making sure k does not exceed population size
        k = min(config.task_params["n_points_to_consider"], len(population) - 1)
        nearest_distances = sorted(distances)[:k]
        novelty_score = sum(nearest_distances) / k
        ind.fitness = novelty_score

    return population
