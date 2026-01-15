#!/usr/bin/env python3
"""
Matplotlib Comparative Plotter for Publication-Quality Figures

This module provides a class for generating publication-quality matplotlib
plots comparing evolutionary computation results across different genotypes
and multiple independent runs. It loads data from SQLite databases and
computes aggregated statistics (mean, standard error) for plotting.

Usage:
    plotter = ComparativePlotter()
    plotter.add_experiment_group(
        name="CPPN",
        db_paths=["path/to/run1.db", "path/to/run2.db", ...],
        color="blue"
    )
    plotter.add_experiment_group(
        name="L-System",
        db_paths=["path/to/lsys_run1.db", ...],
        color="red"
    )
    plotter.plot_average_best_fitness(
        save_path="fitness_comparison.png"
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from sqlmodel import Session, create_engine, select
from rich.console import Console

from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING

console = Console()
type Population = List[Individual]


class ComparativePlotter:
    """
    Publication-quality matplotlib plotter for comparing evolutionary runs.

    This class loads multiple experiment runs from database files, computes
    aggregated statistics across runs, and generates matplotlib figures
    suitable for academic papers.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize the plotter.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.experiment_groups: Dict[str, Dict[str, Any]] = {}

        # Set publication-quality matplotlib parameters
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        })

    def add_experiment_group(
        self,
        name: str,
        db_paths: Optional[List[str]] = None,
        db_directory: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: str = "-",
        marker: Optional[str] = None,
    ) -> None:
        """
        Add an experiment group (e.g., CPPN, L-System, Tree).

        You can either provide:
        - A list of database file paths directly (db_paths)
        - A directory containing multiple .db files (db_directory)

        Args:
            name: Display name for this experiment group
            db_paths: List of paths to database files (one per run)
            db_directory: Directory containing multiple .db files
            color: Color for plotting (matplotlib color string)
            linestyle: Line style (default: '-')
            marker: Marker style (default: None)
        """
        if db_paths is None and db_directory is None:
            raise ValueError("Must provide either db_paths or db_directory")

        if db_paths is None:
            # Load all .db files from directory
            db_dir = Path(db_directory)
            if not db_dir.exists():
                raise FileNotFoundError(f"Directory not found: {db_directory}")
            db_paths = [str(p) for p in sorted(db_dir.glob("*.db"))]
            console.log(
                f"Found {len(db_paths)} database files in {db_directory}"
            )

        if not db_paths:
            raise ValueError(f"No database files found for {name}")

        console.log(f"Loading {name}: {len(db_paths)} runs")

        # Load all runs for this experiment
        runs_data = []
        for db_path in db_paths:
            try:
                populations, config = self._load_populations_from_db(
                    Path(db_path)
                )
                runs_data.append((populations, config))
                console.log(
                    f"  Loaded {db_path}: {len(populations)} generations"
                )
            except Exception as e:
                console.log(f"  Warning: Failed to load {db_path}: {e}")
                continue

        if not runs_data:
            raise ValueError(f"Failed to load any runs for {name}")

        # Compute aggregated statistics
        stats = self._compute_statistics(runs_data)

        self.experiment_groups[name] = {
            "runs_data": runs_data,
            "statistics": stats,
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
            "num_runs": len(runs_data),
        }

        console.log(
            f"Added {name}: {len(runs_data)} runs, "
            f"{stats['max_generation']} generations"
        )

    def _load_populations_from_db(
        self, db_path: Path
    ) -> Tuple[List[Population], Any]:
        """
        Load populations from a single database file.

        Args:
            db_path: Path to the database file

        Returns:
            Tuple of (populations list, config object)
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        engine = create_engine(f"sqlite:///{db_path}")
        populations = []

        with Session(engine) as session:
            # Get all unique generations
            result = session.exec(select(Individual.time_of_birth)).all()
            if not result:
                raise ValueError(f"No individuals found in database: {db_path}")

            generations = sorted(set(result))

            # Load each generation
            for gen in generations:
                individuals = session.exec(
                    select(Individual).where(Individual.time_of_birth == gen)
                ).all()

                population = [
                    Individual(**ind.model_dump()) for ind in individuals
                ]
                populations.append(population)

        # Load configuration
        config = self._load_config(db_path)

        return populations, config

    def _load_config(self, db_path: Path) -> Any:
        """
        Load configuration from JSON file.

        Args:
            db_path: Path to the database file

        Returns:
            Configuration object
        """
        config_filename = db_path.stem + "_config.json"
        json_config_path = db_path.parent / config_filename

        if not json_config_path.exists():
            raise FileNotFoundError(
                f"No configuration file found: {json_config_path}"
            )

        with open(json_config_path, "r") as f:
            config_data = json.load(f)

        resolved = config_data["resolved_settings"]
        genotype_name = resolved["genotype_name"]
        genotype = GENOTYPES_MAPPING[genotype_name]

        class DashboardConfig:
            def __init__(self):
                self.is_maximisation = resolved["is_maximisation"]
                self.first_generation_id = resolved["first_generation_id"]
                self.num_of_generations = resolved["num_of_generations"]
                self.target_population_size = resolved["target_population_size"]
                self.genotype = genotype
                self.task = resolved["task"]
                self.task_params = resolved.get("task_params", {})
                self.genotype_name = genotype_name

        return DashboardConfig()

    def _compute_statistics(
        self, runs_data: List[Tuple[List[Population], Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregated statistics across all runs.

        Args:
            runs_data: List of (populations, config) tuples for each run

        Returns:
            Dictionary containing aggregated statistics
        """
        # Find maximum number of generations across all runs
        max_generation = max(len(populations) for populations, _ in runs_data)
        num_runs = len(runs_data)

        # Initialize arrays to store statistics
        mean_best_fitness = np.zeros(max_generation)
        std_best_fitness = np.zeros(max_generation)
        stderr_best_fitness = np.zeros(max_generation)
        mean_avg_fitness = np.zeros(max_generation)
        std_avg_fitness = np.zeros(max_generation)
        stderr_avg_fitness = np.zeros(max_generation)

        # For each generation, collect best fitness from all runs
        for gen_idx in range(max_generation):
            best_fitnesses = []
            avg_fitnesses = []

            for populations, _ in runs_data:
                if gen_idx < len(populations):
                    population = populations[gen_idx]
                    if population:
                        fitnesses = [ind.fitness for ind in population]
                        best_fitnesses.append(max(fitnesses))
                        avg_fitnesses.append(np.mean(fitnesses))

            if best_fitnesses:
                mean_best_fitness[gen_idx] = np.mean(best_fitnesses)
                std_best_fitness[gen_idx] = np.std(best_fitnesses, ddof=1)
                stderr_best_fitness[gen_idx] = std_best_fitness[
                    gen_idx
                ] / np.sqrt(len(best_fitnesses))

            if avg_fitnesses:
                mean_avg_fitness[gen_idx] = np.mean(avg_fitnesses)
                std_avg_fitness[gen_idx] = np.std(avg_fitnesses, ddof=1)
                stderr_avg_fitness[gen_idx] = std_avg_fitness[
                    gen_idx
                ] / np.sqrt(len(avg_fitnesses))

        return {
            "max_generation": max_generation,
            "num_runs": num_runs,
            "generations": np.arange(max_generation),
            "mean_best_fitness": mean_best_fitness,
            "std_best_fitness": std_best_fitness,
            "stderr_best_fitness": stderr_best_fitness,
            "mean_avg_fitness": mean_avg_fitness,
            "std_avg_fitness": std_avg_fitness,
            "stderr_avg_fitness": stderr_avg_fitness,
        }

    def plot_average_best_fitness(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: str = "Generation",
        ylabel: str = "Best Fitness",
        use_stderr: bool = True,
        show_legend: bool = True,
        legend_loc: str = "best",
        grid: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Plot average best fitness across runs for all experiment groups.

        Args:
            save_path: Path to save the figure (if None, won't save)
            title: Plot title (if None, uses default)
            xlabel: X-axis label
            ylabel: Y-axis label
            use_stderr: If True, use standard error; if False, use std deviation
            show_legend: Whether to show the legend
            legend_loc: Legend location
            grid: Whether to show grid
            show_plot: Whether to display the plot interactively

        Returns:
            The matplotlib Figure object
        """
        if not self.experiment_groups:
            raise ValueError(
                "No experiment groups added. Use add_experiment_group() first."
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each experiment group
        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            generations = stats["generations"]
            mean_fitness = stats["mean_best_fitness"]

            if use_stderr:
                error = stats["stderr_best_fitness"]
                error_label = "SE"
            else:
                error = stats["std_best_fitness"]
                error_label = "SD"

            color = group["color"]
            linestyle = group["linestyle"]
            marker = group["marker"]

            # Plot mean line
            ax.plot(
                generations,
                mean_fitness,
                label=f"{name} (n={group['num_runs']})",
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(generations) // 10),
                zorder=10,
            )

            # Plot shaded error region
            ax.fill_between(
                generations,
                mean_fitness - error,
                mean_fitness + error,
                alpha=0.2,
                color=color,
                zorder=5,
            )

        # Customize plot
        if title is None:
            title = "Average Best Fitness Across Runs"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        if show_legend:
            ax.legend(loc=legend_loc, framealpha=0.9)

        # Tight layout
        fig.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            console.log(f"Saved figure to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_average_fitness(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: str = "Generation",
        ylabel: str = "Average Fitness",
        use_stderr: bool = True,
        show_legend: bool = True,
        legend_loc: str = "best",
        grid: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Plot average population fitness across runs for all experiment groups.

        Similar to plot_average_best_fitness but uses mean fitness instead of best.

        Args:
            save_path: Path to save the figure (if None, won't save)
            title: Plot title (if None, uses default)
            xlabel: X-axis label
            ylabel: Y-axis label
            use_stderr: If True, use standard error; if False, use std deviation
            show_legend: Whether to show the legend
            legend_loc: Legend location
            grid: Whether to show grid
            show_plot: Whether to display the plot interactively

        Returns:
            The matplotlib Figure object
        """
        if not self.experiment_groups:
            raise ValueError(
                "No experiment groups added. Use add_experiment_group() first."
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each experiment group
        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            generations = stats["generations"]
            mean_fitness = stats["mean_avg_fitness"]

            if use_stderr:
                error = stats["stderr_avg_fitness"]
            else:
                error = stats["std_avg_fitness"]

            color = group["color"]
            linestyle = group["linestyle"]
            marker = group["marker"]

            # Plot mean line
            ax.plot(
                generations,
                mean_fitness,
                label=f"{name} (n={group['num_runs']})",
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(generations) // 10),
                zorder=10,
            )

            # Plot shaded error region
            ax.fill_between(
                generations,
                mean_fitness - error,
                mean_fitness + error,
                alpha=0.2,
                color=color,
                zorder=5,
            )

        # Customize plot
        if title is None:
            title = "Average Population Fitness Across Runs"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        if show_legend:
            ax.legend(loc=legend_loc, framealpha=0.9)

        # Tight layout
        fig.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            console.log(f"Saved figure to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_combined(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        use_stderr: bool = True,
        show_legend: bool = True,
        legend_loc: str = "best",
        grid: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Create a combined plot with both best and average fitness.

        Args:
            save_path: Path to save the figure (if None, won't save)
            title: Plot title (if None, uses default)
            use_stderr: If True, use standard error; if False, use std deviation
            show_legend: Whether to show the legend
            legend_loc: Legend location
            grid: Whether to show grid
            show_plot: Whether to display the plot interactively

        Returns:
            The matplotlib Figure object
        """
        if not self.experiment_groups:
            raise ValueError(
                "No experiment groups added. Use add_experiment_group() first."
            )

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5)
        )

        # Plot best fitness (top panel)
        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            generations = stats["generations"]
            mean_fitness = stats["mean_best_fitness"]
            error = (
                stats["stderr_best_fitness"]
                if use_stderr
                else stats["std_best_fitness"]
            )

            ax1.plot(
                generations,
                mean_fitness,
                label=f"{name}",  # (n={group['num_runs']})",
                color=group["color"],
                linestyle=group["linestyle"],
                marker=group["marker"],
                markevery=max(1, len(generations) // 10),
                zorder=10,
            )
            ax1.fill_between(
                generations,
                mean_fitness - error,
                mean_fitness + error,
                alpha=0.2,
                color=group["color"],
                zorder=5,
            )

        ax1.set_title("Best Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best Fitness")
        if grid:
            ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        if show_legend:
            ax1.legend(loc=legend_loc, framealpha=0.9)

        # Plot average fitness (bottom panel)
        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            generations = stats["generations"]
            mean_fitness = stats["mean_avg_fitness"]
            error = (
                stats["stderr_avg_fitness"]
                if use_stderr
                else stats["std_avg_fitness"]
            )

            ax2.plot(
                generations,
                mean_fitness,
                label=f"{name} (n={group['num_runs']})",
                color=group["color"],
                linestyle=group["linestyle"],
                marker=group["marker"],
                markevery=max(1, len(generations) // 10),
                zorder=10,
            )
            ax2.fill_between(
                generations,
                mean_fitness - error,
                mean_fitness + error,
                alpha=0.2,
                color=group["color"],
                zorder=5,
            )

        ax2.set_title("Average Population Fitness")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Average Fitness")
        if grid:
            ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        if title:
            fig.suptitle(title, fontsize=14, y=0.995)

        fig.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            console.log(f"Saved figure to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def get_statistics_summary(self) -> str:
        """
        Get a text summary of loaded statistics.

        Returns:
            Formatted string with statistics summary
        """
        if not self.experiment_groups:
            return "No experiment groups loaded."

        summary = ["=" * 60]
        summary.append("EXPERIMENT GROUPS SUMMARY")
        summary.append("=" * 60)

        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            summary.append(f"\n{name}:")
            summary.append(f"  Number of runs: {stats['num_runs']}")
            summary.append(f"  Generations: {stats['max_generation']}")
            summary.append(f"  Final best fitness (mean ± SE):")
            summary.append(
                f"    {stats['mean_best_fitness'][-1]:.4f} ± "
                f"{stats['stderr_best_fitness'][-1]:.4f}"
            )
            summary.append(f"  Final avg fitness (mean ± SE):")
            summary.append(
                f"    {stats['mean_avg_fitness'][-1]:.4f} ± "
                f"{stats['stderr_avg_fitness'][-1]:.4f}"
            )

        summary.append("=" * 60)
        return "\n".join(summary)


def main():
    """Example usage of the ComparativePlotter class."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from experiment databases"
    )
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        required=True,
        help="Directories containing experiment runs (one per genotype)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Names for each experiment group (e.g., 'CPPN' 'L-System' 'Tree')",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="Colors for each group (matplotlib color strings)",
    )
    parser.add_argument(
        "--output", default="fitness_comparison.png", help="Output file path"
    )
    parser.add_argument(
        "--plot-type",
        choices=["best", "avg", "combined"],
        default="best",
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--use-stderr",
        action="store_true",
        help="Use standard error instead of standard deviation",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    parser.add_argument("--title", default=None, help="Plot title")

    args = parser.parse_args()

    if len(args.experiment_dirs) != len(args.names):
        raise ValueError(
            "Number of experiment directories must match number of names"
        )

    # Default colors if not provided
    if args.colors is None:
        default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        args.colors = default_colors[: len(args.names)]

    # Create plotter
    plotter = ComparativePlotter()

    # Add experiment groups
    for name, exp_dir, color in zip(
        args.names, args.experiment_dirs, args.colors
    ):
        plotter.add_experiment_group(
            name=name, db_directory=exp_dir, color=color
        )

    # Print summary
    print(plotter.get_statistics_summary())

    # Generate plot
    if args.plot_type == "best":
        plotter.plot_average_best_fitness(
            save_path=args.output,
            title=args.title,
            use_stderr=args.use_stderr,
            show_plot=args.show,
        )
    elif args.plot_type == "avg":
        plotter.plot_average_fitness(
            save_path=args.output,
            title=args.title,
            use_stderr=args.use_stderr,
            show_plot=args.show,
        )
    else:  # combined
        plotter.plot_combined(
            save_path=args.output,
            title=args.title,
            use_stderr=args.use_stderr,
            show_plot=args.show,
        )

    console.log(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
