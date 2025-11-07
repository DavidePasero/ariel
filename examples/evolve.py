# Standard library
from __future__ import annotations
import random
from collections.abc import Callable
from pathlib import Path
import tomllib
from typing import Literal
from functools import partial
import matplotlib.pyplot as plt

# Third-party libraries
from fitness_functions import FITNESS_FUNCTIONS
import numpy as np
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EAStep, EA
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
from morphology_fitness_analysis import MorphologyAnalyzer
from ea_settings import EASettings
from evolution_dashboard import run_dashboard

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

# ------------------------ EA STEPS ------------------------ #
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

def read_config_file() -> EASettings:
    cfg = tomllib.loads(Path("examples/config.toml").read_text())

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
    return settings
    

# Example usage
def analyze_evolution_videos(analyzer: MorphologyAnalyzer, populations: list[Population], decoder: Callable) -> None:
    """Create evolution videos for different visualizations."""

    analyzer.create_evolution_video(
        populations=populations,
        decoder=decoder,
        plot_method_name="plot_fitness_distributions",
        video_filename="videos/fitness_distributions.mp4"
    )

    analyzer.create_evolution_video(
        populations=populations,
        decoder=decoder,
        plot_method_name="plot_pairwise_feature_landscapes",
        video_filename="videos/pairwise_feature_landscapes.mp4"
    )


def main() -> None:
    """Entry point."""
    config = read_config_file()
    # Create initial population
    population_list = [create_individual(config) for _ in range(10)]
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
    )

    ea.run()

    best = ea.get_solution(only_alive=False)
    console.log(best)

    median = ea.get_solution("median", only_alive=False)
    console.log(median)

    worst = ea.get_solution("worst", only_alive=False)
    console.log(worst)

    fitnesses = []

    populations = []
    for i in range(1, config.num_of_generations + 1):
        ea.fetch_population(only_alive=False, best_comes=None, custom_logic=[Individual.time_of_birth==i])
        individuals = ea.population
        populations.append(individuals)
        avg_fitness = sum(ind.fitness for ind in individuals) / len(individuals) if individuals else 0
        console.log(f"Generation {i}: Avg Fitness = {avg_fitness}")
        fitnesses.append(avg_fitness)

    # Line plot of the fitness
    plt.plot(range(1, config.num_of_generations + 1), fitnesses, marker='o')
    plt.title(f'{config.genotype.__name__} - {config.task}')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.savefig(f'avg_fitness_over_generations_{config.genotype.__name__}_{config.task}.png')
    plt.show()

    morphology_analyzer = MorphologyAnalyzer()
    morphology_analyzer.load_target_robots(config.task_params["target_robot_path"])

    # analyze_evolution_videos(morphology_analyzer, populations, lambda x: config.genotype.from_json(x).to_digraph())

    # Launch interactive dashboard
    print("\nLaunching Evolution Dashboard...")

    decoder = lambda individual: config.genotype.from_json(individual.genotype).to_digraph()
    run_dashboard(populations, decoder, config)
    

if __name__ == "__main__":
    main()