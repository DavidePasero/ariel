#!/usr/bin/env python3
"""
Dashboard Manager

This module provides a centralized manager for loading databases and
instantiating different dashboard types. It handles all database loading
logic and provides a clean interface for different visualization modes.
"""

import os.path
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent.parent / "src"
sys.path.append(str(src_path))

from typing import List, Tuple, Dict, Any, Optional
import json
import tomllib
from rich.console import Console
from sqlmodel import Session, create_engine, select

from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
import glob

console = Console()
type Population = List[Individual]


class DashboardConfig:
    """Configuration object for dashboards."""

    def __init__(
        self,
        is_maximisation: bool,
        first_generation_id: int,
        num_of_generations: int,
        target_population_size: int,
        genotype: Any,
        task: str,
        task_params: Dict[str, Any],
        genotype_name: str,
        mutation_name: str = "unknown",
        crossover_name: str = "unknown",
        output_folder: Optional[Path] = None,
        db_file_name: Optional[str] = None,
    ):
        """Initialize dashboard configuration."""
        self.is_maximisation = is_maximisation
        self.first_generation_id = first_generation_id
        self.num_of_generations = num_of_generations
        self.target_population_size = target_population_size
        self.genotype = genotype
        self.task = task
        self.task_params = task_params
        self.genotype_name = genotype_name
        self.mutation_name = mutation_name
        self.crossover_name = crossover_name
        self.output_folder = output_folder
        self.db_file_name = db_file_name
        if output_folder and db_file_name:
            self.db_file_path = Path(output_folder) / db_file_name


class DashboardManager:
    """Manager for loading databases and instantiating dashboards."""

    def __init__(self):
        """Initialize the dashboard manager."""
        self.loaded_runs: List[Tuple[List[Population], DashboardConfig]] = []
        self.loaded_genotypes: Dict[
            str, Tuple[List[Population], DashboardConfig]
        ] = {}

    def load_from_database(
        self, db_path: Path, genotype_label: Optional[str] = None
    ) -> Tuple[List[Population], DashboardConfig]:
        """Load populations and config from a single database file.

        Args:
            db_path: Path to the SQLite database file
            genotype_label: Optional label for this genotype

        Returns:
            Tuple of (populations, config)
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        console.log(f"Loading data from: {db_path}")

        # Load populations from database
        engine = create_engine(f"sqlite:///{db_path}")
        populations = []

        with Session(engine) as session:
            result = session.exec(select(Individual.time_of_birth)).all()
            if not result:
                raise ValueError(f"No individuals found in database: {db_path}")

            generations = sorted(set(result))
            console.log(f"Found {len(generations)} generations")

            for gen in generations:
                individuals = session.exec(
                    select(Individual).where(Individual.time_of_birth == gen)
                ).all()

                population = [
                    Individual(**ind.model_dump()) for ind in individuals
                ]
                populations.append(population)

        # Load config - try JSON first, then TOML fallback
        config = self._load_config(db_path)

        return populations, config

    def _load_config(self, db_path: Path) -> DashboardConfig:
        """Load configuration for a database.

        Tries to load from saved JSON config first, then falls back to TOML.

        Args:
            db_path: Path to the database file

        Returns:
            DashboardConfig object
        """
        # Try saved JSON config first
        config_filename = db_path.stem + "_config.json"
        json_config_path = db_path.parent / config_filename

        if json_config_path.exists():
            console.log(f"Loading saved configuration from: {json_config_path}")
            return self._load_config_from_json(json_config_path)

        # Fallback to TOML config
        console.log(
            f"No saved config found at {json_config_path}, trying TOML fallback"
        )
        config_path = db_path.parent / "config.toml"
        if not config_path.exists():
            # Try examples/config.toml as fallback
            config_path = Path("examples/config.toml")

        if config_path.exists():
            console.log(f"Loading TOML configuration from: {config_path}")
            return self._load_config_from_toml(config_path)

        raise FileNotFoundError(
            f"No configuration file found for database {db_path}"
        )

    def _load_config_from_json(self, config_path: Path) -> DashboardConfig:
        """Load configuration from saved JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            DashboardConfig object
        """
        with open(config_path, "r") as f:
            config_data = json.load(f)

        resolved = config_data["resolved_settings"]
        genotype_name = resolved["genotype_name"]
        genotype = GENOTYPES_MAPPING[genotype_name]

        return DashboardConfig(
            is_maximisation=resolved["is_maximisation"],
            first_generation_id=resolved["first_generation_id"],
            num_of_generations=resolved["num_of_generations"],
            target_population_size=resolved["target_population_size"],
            genotype=genotype,
            task=resolved["task"],
            task_params=resolved["task_params"],
            genotype_name=genotype_name,
            mutation_name=resolved.get("mutation_name", "unknown"),
            crossover_name=resolved.get("crossover_name", "unknown"),
        )

    def _load_config_from_toml(self, config_path: Path) -> DashboardConfig:
        """Load configuration from TOML file.

        Args:
            config_path: Path to TOML config file

        Returns:
            DashboardConfig object
        """
        cfg = tomllib.loads(config_path.read_text())

        gname = cfg["run"]["genotype"]
        task = cfg["run"]["task"]
        task_params = cfg["task"].get(task, {}).get("params", {})
        genotype = GENOTYPES_MAPPING[gname]

        return DashboardConfig(
            is_maximisation=cfg["ec"]["is_maximisation"],
            first_generation_id=cfg["ec"]["first_generation_id"],
            num_of_generations=cfg["ec"]["num_of_generations"],
            target_population_size=cfg["ec"]["target_population_size"],
            genotype=genotype,
            task=task,
            task_params=task_params,
            genotype_name=gname,
            output_folder=Path(cfg["data"]["output_folder"]),
            db_file_name=cfg["data"]["db_file_name"],
        )

    def load_multiple_runs(
        self, db_paths: List[Path], run_names: Optional[List[str]] = None
    ) -> None:
        """Load multiple runs from database files.

        Args:
            db_paths: List of paths to database files
            run_names: Optional names for each run
        """
        self.loaded_runs = []

        for i, db_path in enumerate(db_paths):
            try:
                if isinstance(db_path, str):
                    db_path = Path(db_path)
                populations, config = self.load_from_database(db_path)
                self.loaded_runs.append((populations, config))

                run_name = (
                    run_names[i]
                    if run_names and i < len(run_names)
                    else f"Run {i + 1}"
                )
                console.log(
                    f"Loaded {run_name}: {len(populations)} generations"
                )

            except Exception as e:
                console.log(f"[red]Error loading {db_path}: {e}[/red]")
                continue

        if not self.loaded_runs:
            raise ValueError("No runs successfully loaded!")

    def load_multiple_genotypes(
        self, db_paths: List[Path], genotype_names: Optional[List[str]] = None
    ) -> None:
        """Load multiple genotypes from database files.

        Args:
            db_paths: List of paths to database files
            genotype_names: Optional names for each genotype
        """
        self.loaded_genotypes = {}

        for i, db_path in enumerate(db_paths):
            try:
                populations, config = self.load_from_database(db_path)

                # Determine genotype name
                if genotype_names and i < len(genotype_names):
                    genotype_name = genotype_names[i]
                elif hasattr(config, "genotype_name"):
                    genotype_name = config.genotype_name.capitalize()
                else:
                    genotype_name = config.genotype.__name__.replace(
                        "Genotype", ""
                    )

                self.loaded_genotypes[genotype_name] = (populations, config)

                console.log(
                    f"Loaded {genotype_name}: {len(populations)} generations"
                )

                # Log additional config info if available
                if hasattr(config, "mutation_name") and hasattr(
                    config, "crossover_name"
                ):
                    console.log(
                        f"  - Mutation: {config.mutation_name}, Crossover: {config.crossover_name}"
                    )
                if hasattr(config, "task"):
                    console.log(f"  - Task: {config.task}")

            except Exception as e:
                console.log(f"[red]Error loading {db_path}: {e}[/red]")
                continue

        if not self.loaded_genotypes:
            raise ValueError("No genotypes successfully loaded!")

    def create_evolution_dashboard(
        self,
        populations: List[Population],
        decoder: Any,
        config: DashboardConfig,
    ):
        """Create a single-run evolution dashboard.

        Args:
            populations: List of populations per generation
            decoder: Function to decode Individual genotype to robot graph
            config: Dashboard configuration

        Returns:
            EvolutionDashboard instance
        """
        from experiments.genomes.evolution_dashboard import EvolutionDashboard

        return EvolutionDashboard(populations, decoder, config)

    def create_comparative_dashboard(
        self, host: str = "127.0.0.1", port: int = 8051, debug: bool = True
    ):
        """Create a comparative genotypes dashboard from loaded genotypes.

        Args:
            host: Server host address
            port: Server port
            debug: Enable debug mode

        Returns:
            ComparativeEvolutionDashboard instance
        """
        if not self.loaded_genotypes:
            raise ValueError(
                "No genotypes loaded. Call load_multiple_genotypes() first."
            )

        from experiments.genomes.comparative_dashboard import (
            ComparativeEvolutionDashboard,
        )

        console.log(
            f"Creating comparative dashboard with {len(self.loaded_genotypes)} genotypes"
        )
        dashboard = ComparativeEvolutionDashboard(self.loaded_genotypes)
        return dashboard

    def create_multiple_runs_dashboard(
        self,
        run_names: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        port: int = 8052,
        debug: bool = True,
    ):
        """Create a multiple runs dashboard from loaded runs.

        Args:
            run_names: Optional names for each run
            host: Server host address
            port: Server port
            debug: Enable debug mode

        Returns:
            MultipleRunsDashboard instance
        """
        if not self.loaded_runs:
            raise ValueError("No runs loaded. Call load_multiple_runs() first.")

        from experiments.genomes.multiple_runs_dashboard import (
            MultipleRunsDashboard,
        )

        if run_names is None:
            run_names = [f"Run {i + 1}" for i in range(len(self.loaded_runs))]

        console.log(
            f"Creating multiple runs dashboard with {len(self.loaded_runs)} runs"
        )
        dashboard = MultipleRunsDashboard(self.loaded_runs, run_names)
        return dashboard

    def create_novelty_dashboard(
        self,
        populations: List[Population],
        decoder: Any,
        config: DashboardConfig,
    ):
        """Create a single-run novelty search dashboard.

        Args:
            populations: List of populations per generation
            decoder: Function to decode Individual genotype to robot graph
            config: Dashboard configuration

        Returns:
            NoveltySearchDashboard instance
        """
        from experiments.genomes.novelty_dashboard import NoveltySearchDashboard

        return NoveltySearchDashboard(populations, decoder, config)

    def create_novelty_multiple_runs_dashboard(
        self,
        run_names: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        port: int = 8053,
        debug: bool = True,
    ):
        """Create a novelty search multiple runs dashboard from loaded runs.

        Args:
            run_names: Optional names for each run
            host: Server host address
            port: Server port
            debug: Enable debug mode

        Returns:
            MultipleRunsNoveltyDashboard instance
        """
        if not self.loaded_runs:
            raise ValueError("No runs loaded. Call load_multiple_runs() first.")

        from experiments.genomes.multiple_runs_novelty_dashboard import (
            MultipleRunsNoveltyDashboard,
        )

        if run_names is None:
            run_names = [f"Run {i + 1}" for i in range(len(self.loaded_runs))]

        console.log(
            f"Creating novelty search multiple runs dashboard with {len(self.loaded_runs)} runs"
        )
        dashboard = MultipleRunsNoveltyDashboard(self.loaded_runs, run_names)
        return dashboard

    def run_evolution_dashboard(
        self,
        db_path: Path,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = True,
    ):
        """Load and run a single evolution dashboard.

        Args:
            db_path: Path to database file
            host: Server host address
            port: Server port
            debug: Enable debug mode
        """
        populations, config = self.load_from_database(db_path)

        # Define decoder
        def decoder(individual: Individual):
            return config.genotype.from_json(individual.genotype).to_digraph()

        dashboard = self.create_evolution_dashboard(
            populations, decoder, config
        )
        console.log(f"Starting Evolution Dashboard at http://{host}:{port}")
        dashboard.run(host=host, port=port, debug=debug)

    def run_comparative_dashboard(
        self,
        db_paths: List[Path],
        genotype_names: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        port: int = 8051,
        debug: bool = True,
    ):
        """Load and run a comparative dashboard.

        Args:
            db_paths: List of paths to database files (one per genotype)
            genotype_names: Optional names for each genotype
            host: Server host address
            port: Server port
            debug: Enable debug mode
        """
        if len(db_paths) > 3:
            console.log(
                "[yellow]Warning: Only first 3 databases will be used for comparison[/yellow]"
            )
            db_paths = db_paths[:3]

        self.load_multiple_genotypes(db_paths, genotype_names)
        dashboard = self.create_comparative_dashboard(host, port, debug)
        console.log(f"Starting Comparative Dashboard at http://{host}:{port}")
        dashboard.run(host=host, port=port, debug=debug)

    def run_multiple_runs_dashboard(
        self,
        db_paths: List[Path],
        run_names: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        port: int = 8052,
        debug: bool = True,
    ):
        """Load and run a multiple runs dashboard.

        Args:
            db_paths: List of paths to database files (one per run)
            run_names: Optional names for each run
            host: Server host address
            port: Server port
            debug: Enable debug mode
        """
        if os.path.isdir(db_paths[0]):
            db_paths = glob.glob(str(db_paths[0]) + "/*.db")
        self.load_multiple_runs(db_paths, run_names)
        dashboard = self.create_multiple_runs_dashboard(
            run_names, host, port, debug
        )
        console.log(f"Starting Multiple Runs Dashboard at http://{host}:{port}")
        dashboard.run(host=host, port=port, debug=debug)

    def run_novelty_dashboard(
        self,
        db_path: Path,
        host: str = "127.0.0.1",
        port: int = 8052,
        debug: bool = True,
    ):
        """Load and run a single novelty search dashboard.

        Args:
            db_path: Path to database file
            host: Server host address
            port: Server port
            debug: Enable debug mode
        """
        populations, config = self.load_from_database(db_path)

        # Define decoder
        def decoder(individual: Individual):
            return config.genotype.from_json(individual.genotype).to_digraph()

        dashboard = self.create_novelty_dashboard(populations, decoder, config)
        console.log(
            f"Starting Novelty Search Dashboard at http://{host}:{port}"
        )
        dashboard.run(host=host, port=port, debug=debug)

    def run_novelty_multiple_runs_dashboard(
        self,
        db_paths: List[Path],
        run_names: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        port: int = 8053,
        debug: bool = True,
    ):
        """Load and run a novelty search multiple runs dashboard.

        Args:
            db_paths: List of paths to database files (one per run)
            run_names: Optional names for each run
            host: Server host address
            port: Server port
            debug: Enable debug mode
        """
        if os.path.isdir(db_paths[0]):
            db_paths = glob.glob(str(db_paths[0]) + "/*.db")
        self.load_multiple_runs(db_paths, run_names)
        dashboard = self.create_novelty_multiple_runs_dashboard(
            run_names, host, port, debug
        )
        console.log(
            f"Starting Novelty Search Multiple Runs Dashboard at http://{host}:{port}"
        )
        dashboard.run(host=host, port=port, debug=debug)


def main():
    """Main entry point with CLI for dashboard selection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dashboard Manager - Unified interface for evolution dashboards"
    )
    parser.add_argument(
        "--dashboard_type",
        choices=[
            "evolution",
            "comparative",
            "multiple-runs",
            "novelty",
            "novelty-multiple-runs",
        ],
        help="Type of dashboard to launch",
    )
    parser.add_argument(
        "--db_paths",
        nargs="+",
        required=True,
        help="Path(s) to database file(s)",
    )
    parser.add_argument("--names", nargs="+", help="Names for genotypes/runs")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument(
        "--port", type=int, help="Dashboard port (default depends on type)"
    )
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable debug mode"
    )

    args = parser.parse_args()

    # Convert paths
    db_paths = [Path(p) for p in args.db_paths]

    # Create manager
    manager = DashboardManager()

    try:
        if args.dashboard_type == "evolution":
            if len(db_paths) != 1:
                console.log(
                    "[red]Error: Evolution dashboard requires exactly 1 database[/red]"
                )
                return
            port = args.port or 8050
            manager.run_evolution_dashboard(
                db_paths[0], host=args.host, port=port, debug=not args.no_debug
            )

        elif args.dashboard_type == "comparative":
            port = args.port or 8051
            manager.run_comparative_dashboard(
                db_paths,
                genotype_names=args.names,
                host=args.host,
                port=port,
                debug=not args.no_debug,
            )

        elif args.dashboard_type == "multiple-runs":
            port = args.port or 8052
            manager.run_multiple_runs_dashboard(
                db_paths,
                run_names=args.names,
                host=args.host,
                port=port,
                debug=not args.no_debug,
            )

        elif args.dashboard_type == "novelty":
            if len(db_paths) != 1:
                console.log(
                    "[red]Error: Novelty dashboard requires exactly 1 database[/red]"
                )
                return
            port = args.port or 8052
            manager.run_novelty_dashboard(
                db_paths[0], host=args.host, port=port, debug=not args.no_debug
            )

        elif args.dashboard_type == "novelty-multiple-runs":
            port = args.port or 8053
            manager.run_novelty_multiple_runs_dashboard(
                db_paths,
                run_names=args.names,
                host=args.host,
                port=port,
                debug=not args.no_debug,
            )

    except Exception as e:
        console.log(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
