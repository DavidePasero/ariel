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
    genotype = config.genotype.from_json(genotype_json)
    ind_digraph = genotype.to_digraph()
    fitness = config.morphology_analyzer.compute_fitness_score(
        ind_digraph,
        target_graph,
    )
    return fitness


def _eval_single_descriptor(
    genotype_json: str, config: EASettings
) -> list[float]:
    """Helper to compute descriptors for a single individual in parallel."""
    genotype = config.genotype.from_json(genotype_json)
    # Compute the descriptor (CPU intensive)
    return compute_6d_descriptor(genotype.to_digraph())


# -----------------------------------------------


class PreprocessFunc(Protocol):
    def __call__(
        self,
        config: EASettings,
        ea: EA,
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


@register_preprocessing("no_fitness")
def preprocess_no_fitness(
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
        futures = [
            executor.submit(_eval_single_copy, g_json, target_graph, config)
            for g_json in genotypes
        ]

        # Collect results as they complete (in order)
        results = [f.result() for f in futures]

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
    population: Population,
    config: EASettings,
    ea: EA,
    is_fitness_function: bool = True,
) -> Population:
    # 1. Fetch archived population (Serial DB operation)
    ea.fetch_population(
        only_alive=False,
        custom_logic=(
            not Individual.alive,
            Individual.tags_["novel"],
        ),
    )
    old_population = ea.population

    # 2. Combine populations to calculate descriptors in one batch
    all_individuals = old_population + population
    genotypes = [ind.genotype for ind in all_individuals]

    # 3. Parallelize Descriptor Calculation (CPU bound)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_eval_single_descriptor, g_json, config)
            for g_json in genotypes
        ]
        all_descriptors = [f.result() for f in futures]

    # 4. Compute Distance Matrix (Fast enough to run serially)
    distance_matrix = cdist(
        all_descriptors,
        all_descriptors,
        metric="euclidean",
    )

    task_params = (
        config.task_params
        if is_fitness_function
        else config.include_diversity_measure_params
    )

    # 5. Compute novelty scores
    # We must offset indices because distance_matrix includes the archive at the start
    archive_size = len(old_population)

    for idx, ind in enumerate(population):
        # Calculate the actual index in the combined matrix
        matrix_idx = archive_size + idx

        # Get distances for this individual
        distances = distance_matrix[matrix_idx]

        # Exclude self-distance
        distances[matrix_idx] = float("inf")

        # Compute average distance to k nearest neighbors
        # k must not exceed the total number of points available (minus self)
        k = min(task_params["n_points_to_consider"], len(all_individuals) - 1)

        if k > 0:
            nearest_distances = sorted(distances)[:k]
            novelty_score = sum(nearest_distances) / k
        else:
            novelty_score = 0.0

        if is_fitness_function:
            ind.fitness = novelty_score
        else:
            ind.tags["novelty_score"] = novelty_score

    # 6. Archive the top-N novel individuals
    # Instead of a fixed threshold, we select the top 'n_points_to_consider'
    # most novel individuals from the current generation.
    
    n_to_archive = task_params["n_points_to_consider"]
    
    # Helper to retrieve score regardless of storage location
    def get_novelty(ind: Individual) -> float:
        if is_fitness_function:
            return ind.fitness
        return ind.tags.get("novelty_score", 0.0)

    # Sort population descending by novelty score
    sorted_population = sorted(population, key=get_novelty, reverse=True)

    # Tag the top N (or fewer if population is small)
    for i in range(min(len(sorted_population), n_to_archive)):
        sorted_population[i].tags["novel"] = True

    return population


@register_fitness("no_fitness")
def no_fitness(
    population: Population, config: EASettings, ea: EA
) -> Population:
    """_
    Assign 0 fitness to all the individuals to look for biases
    in random generations of a certain genotype

    Args:
        population (Population): _description_
        config (EASettings): _description_
        ea (EA): _description_

    Returns:
        Population: _description_
    """
    for ind in population:
        ind.fitness = 0.0
    return population
