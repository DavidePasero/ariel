#!/usr/bin/env python3
"""
Dashboard Loader

Loads evolution data from SQLite database and displays interactive dashboard.
Run this after headless evolution to visualize results.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import tomllib
import json
from typing import Any

from rich.console import Console
from sqlmodel import Session, create_engine, select

from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
from evolution_dashboard import run_dashboard

console = Console()

type Population = list[Individual]


def load_config_from_saved_json(config_path: Path) -> Any:
    """Load configuration from saved JSON file created by evolve_headless."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Extract the resolved settings
    resolved = config_data["resolved_settings"]
    genotype_name = resolved["genotype_name"]

    # Get the genotype class
    genotype = GENOTYPES_MAPPING[genotype_name]

    # Create a config object with necessary attributes
    class DashboardConfig:
        def __init__(self):
            self.is_maximisation = resolved["is_maximisation"]
            self.first_generation_id = resolved["first_generation_id"]
            self.num_of_generations = resolved["num_of_generations"]
            self.target_population_size = resolved["target_population_size"]
            self.genotype = genotype
            self.task = resolved["task"]
            self.task_params = resolved["task_params"]
            self.genotype_name = genotype_name
            self.output_folder = Path(resolved["output_folder"])
            self.db_file_name = resolved["db_file_name"]
            self.db_file_path = Path(resolved["output_folder"]) / resolved["db_file_name"]
            # Store the full config data for reference
            self._full_config = config_data

    return DashboardConfig()


def load_config() -> Any:
    """Load configuration from config.toml (fallback)."""
    cfg = tomllib.loads(Path("examples/config.toml").read_text())

    # Basic configuration needed for dashboard
    gname = cfg["run"]["genotype"]
    task = cfg["run"]["task"]
    task_params = cfg["task"].get(task, {}).get("params", {})

    genotype = GENOTYPES_MAPPING[gname]

    # Create a simple config object for dashboard (avoiding EASettings validation)
    class DashboardConfig:
        def __init__(self):
            self.is_maximisation = cfg["ec"]["is_maximisation"]
            self.first_generation_id = cfg["ec"]["first_generation_id"]
            self.num_of_generations = cfg["ec"]["num_of_generations"]
            self.target_population_size = cfg["ec"]["target_population_size"]
            self.genotype = genotype
            self.task = task
            self.task_params = task_params
            self.genotype_name = gname
            self.output_folder = Path(cfg["data"]["output_folder"])
            self.db_file_name = cfg["data"]["db_file_name"]
            self.db_file_path = Path(cfg["data"]["output_folder"]) / cfg["data"]["db_file_name"]

    return DashboardConfig()


def load_populations_from_database(db_path: Path) -> tuple[list[Population], Any]:
    """Load all populations from the database, grouped by generation."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    console.log(f"Loading evolution data from: {db_path}")

    # Connect to database
    engine = create_engine(f"sqlite:///{db_path}")

    populations = []

    with Session(engine) as session:
        # Find the range of generations
        result = session.exec(select(Individual.time_of_birth)).all()
        if not result:
            raise ValueError("No individuals found in database")

        generations = sorted(set(result))
        console.log(f"Found {len(generations)} generations: {min(generations)} to {max(generations)}")

        # Load each generation
        for gen in generations:
            individuals = session.exec(
                select(Individual).where(Individual.time_of_birth == gen)
            ).all()

            # Convert to list to avoid SQLModel session issues
            population = [Individual.model_validate(ind.model_dump()) for ind in individuals]
            populations.append(population)

            console.log(f"Generation {gen}: {len(population)} individuals")

    # Try to load the saved JSON config first
    config_filename = db_path.stem + "_config.json"
    json_config_path = db_path.parent / config_filename

    if json_config_path.exists():
        console.log(f"Loading saved configuration from: {json_config_path}")
        config = load_config_from_saved_json(json_config_path)
    else:
        console.log(f"No saved config found at {json_config_path}, using fallback TOML config")
        config = load_config()

    return populations, config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load evolution dashboard from database")
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to database file (default: from config.toml)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    # Determine database path
    if args.db_path:
        db_path = args.db_path
    else:
        config = load_config()
        db_path = config.db_file_path

    try:
        # Load evolution data from database
        populations, config = load_populations_from_database(db_path)

        if not populations:
            console.log("No populations found in database!")
            return

        console.log(f"Successfully loaded {len(populations)} generations")

        # Create decoder function for genotype visualization
        def decoder(individual: Individual):
            return config.genotype.from_json(individual.genotype).to_digraph()

        console.log(f"Starting dashboard at http://{args.host}:{args.port}")
        console.log("Press Ctrl+C to stop")

        # Launch dashboard
        run_dashboard(
            populations=populations,
            decoder=decoder,
            config=config,
            host=args.host,
            port=args.port,
            debug=not args.no_debug
        )

    except Exception as e:
        console.log(f"Error loading dashboard: {e}")
        raise


if __name__ == "__main__":
    main()
