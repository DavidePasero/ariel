from typing import Callable, Dict, Protocol
from pathlib import Path
from scipy.spatial.distance import cdist

from ea_settings import EASettings
from ariel.ec.a001 import Individual

from morphology_fitness_analysis import compute_6d_descriptor, load_target_robot, compute_fitness_scores

type Population = list[Individual]


# Types
class FitnessFunc(Protocol):
    def __call__(self, population: Population, config: EASettings) -> Population: ...

FITNESS_FUNCTIONS: Dict[str, FitnessFunc] = {}

def register_fitness(name: str | None = None):
    def deco(fn: FitnessFunc) -> FitnessFunc:
        FITNESS_FUNCTIONS[name or fn.__name__] = fn
        return fn
    return deco

def _get_6d_descriptors(individual: Individual, config: EASettings):
    genotype = config.genotype.from_json(individual.genotype)
    # Convert to digraph
    ind_digraph = genotype.to_digraph()
    # Compute the morphological descriptors
    measures = compute_6d_descriptor(ind_digraph)
    return measures

@register_fitness("evolve_to_copy")
def evolve_to_copy(population: Population, config: EASettings) -> Population:
    target_descriptor = load_target_robot(Path(config.task_params["target_robot_path"]))
    for ind in population:
        measures = _get_6d_descriptors(ind, config)
        fitness = compute_fitness_scores(target_descriptor, measures)
        ind.fitness = fitness
    return population

@register_fitness("evolve_for_novelty")
def evolve_for_novelty(population: Population, config: EASettings) -> Population:
    robot_descriptors = []
    for ind in population:
        measures = _get_6d_descriptors(ind, config)
        robot_descriptors.append(measures)

    distance_matrix = cdist(robot_descriptors, robot_descriptors, metric='euclidean')
    for idx, ind in enumerate(population):
        # Exclude self-distance by setting it to infinity
        distances = distance_matrix[idx]
        distances[idx] = float('inf')
        # Compute average distance to k nearest neighbors, making sure k does not exceed population size
        k = min(config.task_params["n_points_to_consider"], len(population) - 1)
        nearest_distances = sorted(distances)[:k]
        novelty_score = sum(nearest_distances) / k
        ind.fitness = novelty_score

    return population