#!/usr/bin/env python3
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Import the refactored metrics
from metrics import (
    calculate_similarity_descriptor,
    calculate_similarity_ted,
    compute_6d_descriptor,
)
from sklearn.decomposition import PCA

from ariel.ec.a001 import Individual

# Import ARIEL and custom metrics
from ariel.utils.graph_ops import load_robot_json_file

type Population = list[Individual]

# Set random seed
np.random.seed(42)


class MorphologyAnalyzer:
    """Analyze and visualize morphological fitness landscapes."""

    def __init__(self, metric: Literal["descriptor", "ted"] = "descriptor"):
        self.target_graphs = []  # Store raw graphs for TED calculation
        self.target_descriptors = []
        self.target_names = []
        self.descriptors = []

        # Store graphs of the random population for TED calculation
        self.fitness_scores = []
        self.metric = metric

    def load_target_robots(self, *json_paths: str):
        """Load target robots, store graphs and compute descriptors."""
        self.target_descriptors = []
        self.target_names = []
        self.target_graphs = []

        for json_path in json_paths:
            try:
                robot_graph = load_robot_json_file(json_path)
                descriptor = compute_6d_descriptor(robot_graph)

                self.target_graphs.append(robot_graph)
                self.target_descriptors.append(descriptor)

                name = Path(json_path).stem
                self.target_names.append(name)
                print(f"Loaded {name}: {descriptor}")

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        self.target_descriptors = np.array(self.target_descriptors)

    def compute_fitness_score(
        self,
        individual: nx.DiGraph,
        target_graph: nx.DiGraph,
    ) -> float:
        """
        Compute fitness score for a single individual against a single target.
        """

        if self.metric == "descriptor":
            distance = calculate_similarity_descriptor(individual, target_graph)
        elif self.metric == "ted":
            distance = calculate_similarity_ted(individual, target_graph)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Return negative distance (maximization problem)
        return -distance

    def plot_fitness_landscapes(self, return_fig: bool = False) -> None:
        """Plot fitness landscapes. Position is PCA of descriptors, Color is Fitness."""
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        # PCA is ALWAYS based on descriptors to provide the "Map"
        all_desc = np.vstack([self.target_descriptors, self.descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[: len(self.target_descriptors)]
        evolved_pca = all_pca[len(self.target_descriptors) :]

        n_targets = len(self.target_names)
        fig, axes = plt.subplots(2, (n_targets + 1) // 2, figsize=(15, 10))
        if n_targets == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (target_name, fitness) in enumerate(
            zip(self.target_names, self.fitness_scores)
        ):
            ax = axes[i]
            scatter = ax.scatter(
                evolved_pca[:, 0],
                evolved_pca[:, 1],
                c=fitness,
                cmap="viridis",
                alpha=0.6,
                s=30,
            )
            ax.scatter(
                target_pca[i, 0],
                target_pca[i, 1],
                c="red",
                s=200,
                marker="*",
                edgecolors="black",
                label=f"Target: {target_name}",
            )
            ax.set_title(f"Fitness Landscape: {target_name}")
            plt.colorbar(scatter, ax=ax, label="Fitness (-Distance)")

        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()
        return None


def main():
    target_paths = [
        "examples/target_robots/small_robot_8.json",
        "examples/target_robots/medium_robot_15.json",
    ]

    analyzer = MorphologyAnalyzer()
    analyzer.load_target_robots(*target_paths)

    # Generate smaller population for TED demo (it is slower)
    analyzer.generate_random_population(n_robots=20)

    # 1. Analyze using Descriptors
    print("\n--- Analysis with Morphological Descriptors ---")
    analyzer.compute_fitness_scores(metric="descriptor")
    analyzer.plot_fitness_landscapes()

    # 2. Analyze using Tree Edit Distance
    print("\n--- Analysis with Tree Edit Distance ---")
    analyzer.compute_fitness_scores(metric="ted")
    analyzer.plot_fitness_landscapes()


if __name__ == "__main__":
    main()
