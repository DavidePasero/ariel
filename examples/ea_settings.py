from __future__ import annotations
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

# Third-party libraries
from pydantic_settings import BaseSettings

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.mutations import Mutation
from ariel.ec.crossovers import Crossover
from ariel.ec.genotypes.genotype import Genotype

# Type Aliases
type Population = list[Individual]
type PopulationFunc = Callable[[Population], Population]
DB_HANDLING_MODES = Literal["delete", "halt"]

class EASettings(BaseSettings):
    quiet: bool = False

    # EC mechanisms
    is_maximisation: bool = True
    first_generation_id: int = 0
    num_of_generations: int = 100
    target_population_size: int = 100
    genotype: type[Genotype]
    create_individual_params: dict = {}
    mutation: Mutation
    mutation_params: dict = {}
    crossover: Crossover
    crossover_params: dict = {}

    task: str = "evolve_to_copy"
    task_params: dict = {}

    # Data config
    output_folder: Path = Path.cwd() / "__data__"
    db_file_name: str = "database.db"
    db_file_path: Path = output_folder / db_file_name
    db_handling: DB_HANDLING_MODES = "delete"