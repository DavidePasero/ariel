#!/usr/bin/env python3
"""
Unified Comparative Plotter (Fitness or Novelty)

This module provides a class for generating publication-quality matplotlib
plots comparing evolutionary computation results. It can switch between
standard Fitness plots and Novelty Search plots via command line arguments.
It also reports the overall best metrics found per experiment group.

Usage:
    python plotter.py --type fitness --experiment-dirs ...
    python plotter.py --type novelty --experiment-dirs ...
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from sqlmodel import Session, create_engine, select
from rich.console import Console
from rich.table import Table

from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING

console = Console()
type Population = List[Individual]


class ComparativePlotter:
    """
    Publication-quality matplotlib plotter for comparing evolutionary runs.
    Supports switching between Fitness and Novelty metrics.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300,
        metric_type: str = "fitness",
    ):
        """
        Initialize the plotter.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: DPI for saved figures
            metric_type: 'fitness' or 'novelty'
        """
        self.figsize = figsize
        self.dpi = dpi
        self.metric_type = metric_type
        self.experiment_groups: Dict[str, Dict[str, Any]] = {}

        # Set labels based on metric type
        if self.metric_type == "novelty":
            self.label_best = "Max Novelty Score"
            self.label_avg = "Average Novelty Score"
            self.label_y = "Novelty Score"
        else:
            self.label_best = "Best Fitness"
            self.label_avg = "Average Fitness"
            self.label_y = "Fitness"

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
        """Add an experiment group (e.g., CPPN, L-System)."""
        if db_paths is None and db_directory is None:
            raise ValueError("Must provide either db_paths or db_directory")

        if db_paths is None:
            db_dir = Path(db_directory)
            if not db_dir.exists():
                raise FileNotFoundError(f"Directory not found: {db_directory}")
            db_paths = [str(p) for p in sorted(db_dir.glob("*.db"))]
            console.log(
                f"Found {len(db_paths)} database files in {db_directory}"
            )

        if not db_paths:
            raise ValueError(f"No database files found for {name}")

        console.log(
            f"Loading {name} ({self.metric_type} mode): {len(db_paths)} runs"
        )

        runs_data = []
        for db_path in db_paths:
            try:
                populations, config = self._load_populations_from_db(
                    Path(db_path)
                )
                runs_data.append((populations, config))
            except Exception as e:
                console.log(
                    f"  [red]Warning[/red]: Failed to load {db_path}: {e}"
                )
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

    def _load_populations_from_db(
        self, db_path: Path
    ) -> Tuple[List[Population], Any]:
        """Load populations from a single database file."""
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        engine = create_engine(f"sqlite:///{db_path}")
        populations = []

        with Session(engine) as session:
            result = session.exec(select(Individual.time_of_birth)).all()
            if not result:
                raise ValueError(f"No individuals found in database: {db_path}")

            generations = sorted(set(result))

            for gen in generations:
                individuals = session.exec(
                    select(Individual).where(Individual.time_of_birth == gen)
                ).all()

                population = [
                    Individual(**ind.model_dump()) for ind in individuals
                ]
                populations.append(population)

        # Try loading config, return None if missing (not strictly needed for plotting)
        try:
            config = self._load_config(db_path)
        except Exception:
            config = None

        return populations, config

    def _load_config(self, db_path: Path) -> Any:
        """Load configuration from JSON file."""
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
                self.genotype = genotype
                self.genotype_name = genotype_name

        return DashboardConfig()

    def _compute_statistics(
        self, runs_data: List[Tuple[List[Population], Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregated statistics based on the selected metric_type.
        """
        max_generation = max(len(populations) for populations, _ in runs_data)
        num_runs = len(runs_data)

        # Initialize arrays
        mean_best = np.zeros(max_generation)
        std_best = np.zeros(max_generation)
        stderr_best = np.zeros(max_generation)
        mean_avg = np.zeros(max_generation)
        std_avg = np.zeros(max_generation)
        stderr_avg = np.zeros(max_generation)

        # Also store per-run bests for final reporting
        run_bests = []

        for populations, _ in runs_data:
            # Find the absolute best value reached in this specific run
            run_max_val = -float("inf")

            for population in populations:
                if not population:
                    continue

                if self.metric_type == "novelty":
                    vals = []
                    for ind in population:
                        val = ind.tags.get("novelty_score")
                        if val is None:
                            val = ind.tags.get("novelty", 0.0)
                        vals.append(float(val))
                else:
                    vals = [ind.fitness for ind in population]

                if vals:
                    current_max = max(vals)
                    if current_max > run_max_val:
                        run_max_val = current_max

            run_bests.append(run_max_val)

        for gen_idx in range(max_generation):
            best_values = []
            avg_values = []

            for populations, _ in runs_data:
                if gen_idx < len(populations):
                    population = populations[gen_idx]
                    if population:
                        # --- DATA EXTRACTION SWITCH ---
                        if self.metric_type == "novelty":
                            # Extract novelty_score from tags
                            values = []
                            for ind in population:
                                # Safe get, default to 0.0 if missing to avoid crashes
                                val = ind.tags.get("novelty_score")
                                if val is None:
                                    # Fallback if using 'novelty' key instead of 'novelty_score'
                                    val = ind.tags.get("novelty", 0.0)
                                values.append(float(val))
                        else:
                            # Standard fitness
                            values = [ind.fitness for ind in population]
                        # ------------------------------

                        if values:
                            best_values.append(max(values))
                            avg_values.append(np.mean(values))

            # Helper to calculate stats safely
            def calc_stat(data_list, out_mean, out_std, out_stderr, idx):
                if data_list:
                    out_mean[idx] = np.mean(data_list)
                    out_std[idx] = np.std(data_list, ddof=1)
                    out_stderr[idx] = out_std[idx] / np.sqrt(len(data_list))

            calc_stat(best_values, mean_best, std_best, stderr_best, gen_idx)
            calc_stat(avg_values, mean_avg, std_avg, stderr_avg, gen_idx)

        return {
            "max_generation": max_generation,
            "num_runs": num_runs,
            "generations": np.arange(max_generation),
            "mean_best": mean_best,
            "std_best": std_best,
            "stderr_best": stderr_best,
            "mean_avg": mean_avg,
            "std_avg": std_avg,
            "stderr_avg": stderr_avg,
            "run_bests": run_bests,  # Store per-run bests
        }

    def print_best_results(self):
        """Prints a rich table with the best results per group."""
        table = Table(
            title=f"Best Results Summary ({self.metric_type.capitalize()})"
        )
        table.add_column("Experiment Group", style="cyan", no_wrap=True)
        table.add_column("Runs", justify="center")
        table.add_column("Best Overall", justify="right", style="green")
        table.add_column("Mean Best (± Std)", justify="right")
        table.add_column("Worst Best", justify="right", style="red")

        for name, group in self.experiment_groups.items():
            run_bests = group["statistics"]["run_bests"]
            # Filter out -inf if any runs failed completely
            valid_bests = [x for x in run_bests if x != -float("inf")]

            if not valid_bests:
                table.add_row(name, str(len(run_bests)), "N/A", "N/A", "N/A")
                continue

            best_overall = np.max(valid_bests)
            mean_best = np.mean(valid_bests)
            std_best = np.std(valid_bests)
            worst_best = np.min(valid_bests)

            table.add_row(
                name,
                str(len(run_bests)),
                f"{best_overall:.4f}",
                f"{mean_best:.4f} ± {std_best:.4f}",
                f"{worst_best:.4f}",
            )

        console.print(table)
        console.print("")  # spacing

    def plot_average_best(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        use_stderr: bool = True,
        show_legend: bool = True,
        grid: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Plot the 'Best' metric (Max Fitness or Max Novelty) across runs.
        """
        if not self.experiment_groups:
            raise ValueError("No experiment groups added.")

        fig, ax = plt.subplots(figsize=self.figsize)

        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            generations = stats["generations"]
            mean_val = stats["mean_best"]
            error = stats["stderr_best"] if use_stderr else stats["std_best"]

            ax.plot(
                generations,
                mean_val,
                label=f"{name}",
                color=group["color"],
                linestyle=group["linestyle"],
                marker=group["marker"],
                markevery=max(1, len(generations) // 10),
            )
            ax.fill_between(
                generations,
                mean_val - error,
                mean_val + error,
                alpha=0.2,
                color=group["color"],
                linewidth=0,
            )

        if title is None:
            title = f"{self.label_best} Across Runs"

        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(
            self.label_best
        )  # E.g., "Max Novelty Score" or "Best Fitness"

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        if show_legend:
            ax.legend(loc="best", framealpha=0.9)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            console.log(f"Saved figure to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_average_mean(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        use_stderr: bool = True,
        show_legend: bool = True,
        grid: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Plot the 'Average' metric (Mean Fitness or Mean Novelty) across runs.
        """
        if not self.experiment_groups:
            raise ValueError("No experiment groups added.")

        fig, ax = plt.subplots(figsize=self.figsize)

        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            generations = stats["generations"]
            mean_val = stats["mean_avg"]
            error = stats["stderr_avg"] if use_stderr else stats["std_avg"]

            ax.plot(
                generations,
                mean_val,
                label=f"{name}",
                color=group["color"],
                linestyle=group["linestyle"],
                marker=group["marker"],
                markevery=max(1, len(generations) // 10),
            )
            ax.fill_between(
                generations,
                mean_val - error,
                mean_val + error,
                alpha=0.2,
                color=group["color"],
                linewidth=0,
            )

        if title is None:
            title = f"{self.label_avg} Across Runs"

        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(self.label_avg)

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        if show_legend:
            ax.legend(loc="best", framealpha=0.9)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            console.log(f"Saved figure to: {save_path}")

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
    ) -> plt.Figure:
        """Create a combined plot with both Best and Average metrics."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5)
        )

        # Plot Best (Top Panel)
        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            gens = stats["generations"]
            mean_val = stats["mean_best"]
            error = stats["stderr_best"] if use_stderr else stats["std_best"]

            ax1.plot(
                gens,
                mean_val,
                label=name,
                color=group["color"],
                linestyle=group["linestyle"],
            )
            ax1.fill_between(
                gens,
                mean_val - error,
                mean_val + error,
                alpha=0.2,
                color=group["color"],
            )

        ax1.set_title(self.label_best)
        ax1.set_ylabel(self.label_y)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(loc="best")

        # Plot Average (Bottom Panel)
        for name, group in self.experiment_groups.items():
            stats = group["statistics"]
            gens = stats["generations"]
            mean_val = stats["mean_avg"]
            error = stats["stderr_avg"] if use_stderr else stats["std_avg"]

            ax2.plot(
                gens,
                mean_val,
                label=name,
                color=group["color"],
                linestyle=group["linestyle"],
            )
            ax2.fill_between(
                gens,
                mean_val - error,
                mean_val + error,
                alpha=0.2,
                color=group["color"],
            )

        ax2.set_title(self.label_avg)
        ax2.set_ylabel(self.label_y)
        ax2.set_xlabel("Generation")
        ax2.grid(True, alpha=0.3, linestyle="--")

        if title:
            fig.suptitle(title, fontsize=14, y=0.995)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            console.log(f"Saved combined figure to: {save_path}")

        return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comparative plots for Fitness OR Novelty."
    )

    parser.add_argument(
        "--type",
        choices=["fitness", "novelty"],
        default="fitness",
        help="Metric to plot: 'fitness' uses ind.fitness, 'novelty' uses ind.tags['novelty_score'].",
    )

    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        required=True,
        help="Dirs containing .db files",
    )
    parser.add_argument(
        "--names", nargs="+", required=True, help="Names for each group"
    )
    parser.add_argument(
        "--colors", nargs="+", default=None, help="Colors for each group"
    )
    parser.add_argument(
        "--output", default="comparison.png", help="Output file path"
    )

    # Plot modes (best/avg/combined)
    parser.add_argument(
        "--plot-mode",
        choices=["best", "avg", "combined"],
        default="best",
        help="Which statistic to plot",
    )

    parser.add_argument(
        "--use-stderr", action="store_true", help="Use standard error bars"
    )
    parser.add_argument("--title", default=None, help="Plot title")

    args = parser.parse_args()

    if len(args.experiment_dirs) != len(args.names):
        raise ValueError("Count of directories must match count of names")

    # Default colors
    if args.colors is None:
        args.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        # Cycle if needed
        while len(args.colors) < len(args.names):
            args.colors += args.colors

    # Initialize with the specific type
    plotter = ComparativePlotter(metric_type=args.type)

    for name, exp_dir, color in zip(
        args.names, args.experiment_dirs, args.colors
    ):
        plotter.add_experiment_group(
            name=name, db_directory=exp_dir, color=color
        )

    # --- PRINT BEST RESULTS ---
    plotter.print_best_results()
    # --------------------------

    # Generate requested plot
    if args.plot_mode == "best":
        plotter.plot_average_best(
            save_path=args.output,
            title=args.title,
            use_stderr=args.use_stderr,
        )
    elif args.plot_mode == "avg":
        plotter.plot_average_mean(
            save_path=args.output,
            title=args.title,
            use_stderr=args.use_stderr,
        )
    else:
        plotter.plot_combined(
            save_path=args.output,
            title=args.title,
            use_stderr=args.use_stderr,
        )


if __name__ == "__main__":
    main()
