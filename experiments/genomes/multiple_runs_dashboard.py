#!/usr/bin/env python3
"""
Multiple Runs Dashboard

This module provides an interactive web dashboard for analyzing evolutionary
computation results across multiple independent runs using Plotly Dash. It
displays aggregated statistics (mean, std) across runs for fitness, morphological
features, and diversity metrics.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
import json
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Any
from rich.console import Console
from sqlmodel import Session, create_engine, select
from networkx.readwrite import json_graph
import mujoco as mj

from experiments.genomes.plotly_morphology_analysis import (
    PlotlyMorphologyAnalyzer,
)
from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer

console = Console()
type Population = List[Individual]


class MultipleRunsDashboard:
    """Interactive dashboard for analyzing multiple evolutionary runs."""

    def __init__(
        self,
        runs_data: List[Tuple[List[Population], Any]],
        run_names: List[str] = None,
    ):
        """Initialize dashboard with multiple run data.

        Args:
            runs_data: List of (populations, config) tuples, one per run
            run_names: Optional names for each run (default: "Run 1", "Run 2", ...)
        """
        self.runs_data = runs_data
        self.num_runs = len(runs_data)

        if run_names is None:
            self.run_names = [f"Run {i + 1}" for i in range(self.num_runs)]
        else:
            self.run_names = run_names

        # Use the first run's config as the reference
        _, self.config = runs_data[0]

        # Initialize analyzer (shared for all runs, same genotype)
        self.analyzer = PlotlyMorphologyAnalyzer()

        # Load target robots if available
        if (
            hasattr(self.config, "task_params")
            and "target_robot_path" in self.config.task_params
        ):
            self.analyzer.load_target_robots(
                str(self.config.task_params["target_robot_path"])
            )

        # Pre-compute aggregated fitness statistics
        self._compute_aggregated_fitness()

        # Cache for computed descriptors per run per generation
        self._descriptor_cache = {}

        # Cache for individual data (for click handling)
        self._individual_cache = {}

        # Morphological feature names
        self.feature_names = [
            "Branching",
            "Limbs",
            "Extensiveness",
            "Symmetry",
            "Proportion",
            "Joints",
        ]

        # Create dl_robots directory if it doesn't exist
        self.dl_robots_path = Path("examples/dl_robots")
        self.dl_robots_path.mkdir(exist_ok=True)

        # Store for robot visualization
        self.current_robot_image = None

        # Load target robot graphs for rendering
        self._load_target_robot_graphs()

        # Color scheme for individual runs
        self.run_colors = px.colors.qualitative.Set2[: self.num_runs]

        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()

    def _color_to_rgba(self, color: str, alpha: float = 0.3) -> str:
        """Convert any color format to rgba string with opacity.

        Args:
            color: Color in hex or rgb format
            alpha: Opacity value (0-1)

        Returns:
            RGBA color string
        """
        if color.startswith("#"):
            # Hex color
            rgb = px.colors.hex_to_rgb(color)
            return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"
        elif color.startswith("rgb"):
            # Already in rgb/rgba format
            # Extract RGB values
            import re

            match = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
            if match:
                r, g, b = match.groups()
                return f"rgba({r},{g},{b},{alpha})"
        # Fallback
        return f"rgba(100,100,100,{alpha})"

    def _load_target_robot_graphs(self):
        """Load target robot graphs from JSON files."""
        self.target_robot_graphs = {}

        if (
            hasattr(self.config, "task_params")
            and "target_robot_path" in self.config.task_params
        ):
            target_path = Path(self.config.task_params["target_robot_path"])

            if target_path.is_file():
                target_files = [target_path]
            elif target_path.is_dir():
                target_files = sorted(target_path.glob("*.json"))
            else:
                target_files = []

            for target_file in target_files:
                try:
                    with open(target_file, "r") as f:
                        robot_data = json.load(f)
                        robot_graph = json_graph.node_link_graph(
                            robot_data, edges="edges"
                        )
                        robot_name = target_file.stem
                        self.target_robot_graphs[robot_name] = robot_graph
                except Exception as e:
                    print(
                        f"Warning: Could not load target robot {target_file}: {e}"
                    )

    def _get_target_robot_options(self):
        """Get dropdown options for target robots."""
        return [
            {"label": name, "value": name}
            for name in self.target_robot_graphs.keys()
        ]

    def _compute_aggregated_fitness(self):
        """Compute aggregated fitness statistics across all runs."""
        # Find max number of generations across all runs
        self.max_generation = max(
            len(populations) for populations, _ in self.runs_data
        )

        # Store individual run timelines for overlay plots
        self.individual_run_timelines = []

        for run_idx, (populations, config) in enumerate(self.runs_data):
            timeline = []
            for gen_idx, population in enumerate(populations):
                if population:
                    fitnesses = [ind.fitness for ind in population]
                    timeline.append({
                        "generation": gen_idx,
                        "avg_fitness": np.mean(fitnesses),
                        "std_fitness": np.std(fitnesses),
                        "best_fitness": max(fitnesses),
                        "worst_fitness": min(fitnesses),
                        "median_fitness": np.median(fitnesses),
                    })
            self.individual_run_timelines.append(timeline)

        # Compute aggregated statistics across runs
        self.aggregated_timeline = []

        for gen_idx in range(self.max_generation):
            # Collect fitness values from all runs at this generation
            avg_fitnesses = []
            best_fitnesses = []
            worst_fitnesses = []
            median_fitnesses = []

            for timeline in self.individual_run_timelines:
                if gen_idx < len(timeline):
                    avg_fitnesses.append(timeline[gen_idx]["avg_fitness"])
                    best_fitnesses.append(timeline[gen_idx]["best_fitness"])
                    worst_fitnesses.append(timeline[gen_idx]["worst_fitness"])
                    median_fitnesses.append(timeline[gen_idx]["median_fitness"])

            if avg_fitnesses:
                self.aggregated_timeline.append({
                    "generation": gen_idx,
                    "mean_avg_fitness": np.mean(avg_fitnesses),
                    "std_avg_fitness": np.std(avg_fitnesses),
                    "mean_best_fitness": np.mean(best_fitnesses),
                    "std_best_fitness": np.std(best_fitnesses),
                    "mean_worst_fitness": np.mean(worst_fitnesses),
                    "std_worst_fitness": np.std(worst_fitnesses),
                    "mean_median_fitness": np.mean(median_fitnesses),
                    "overall_best": max(best_fitnesses),
                    "overall_worst": min(worst_fitnesses),
                })

    def _get_generation_data(self, run_idx: int, generation: int):
        """Get or compute morphological data for a specific run and generation."""
        cache_key = f"run{run_idx}_gen{generation}"

        if cache_key in self._descriptor_cache:
            return self._descriptor_cache[cache_key]

        populations, config = self.runs_data[run_idx]

        if generation >= len(populations):
            generation = len(populations) - 1

        # Load population data into analyzer
        population = populations[generation]

        def decoder(individual: Individual):
            return config.genotype.from_json(individual.genotype).to_digraph()

        self.analyzer.load_population(population, decoder)
        self.analyzer.compute_fitness_scores()

        # Cache the results
        self._descriptor_cache[cache_key] = {
            "descriptors": self.analyzer.descriptors.copy(),
            "fitness_scores": self.analyzer.fitness_scores.copy(),
        }

        # Cache individual data for click handling
        self._individual_cache[cache_key] = {
            "population": population.copy(),
            "decoder": decoder,
            "run_idx": run_idx,
        }

        return self._descriptor_cache[cache_key]

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1(
                "Multiple Runs Analysis Dashboard",
                style={"textAlign": "center", "marginBottom": 10},
            ),
            html.Div([
                html.P(
                    f"Analyzing {self.num_runs} independent evolutionary runs",
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "marginBottom": 30,
                    },
                )
            ]),
            # Status message area
            html.Div(
                id="status-message",
                style={
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "backgroundColor": "#f0f0f0",
                    "borderRadius": "5px",
                    "display": "none",
                },
            ),
            # Generation control section
            html.Div(
                [
                    html.Label(
                        "Select Generation:",
                        style={"fontWeight": "bold", "marginBottom": 10},
                    ),
                    dcc.Slider(
                        id="generation-slider",
                        min=0,
                        max=self.max_generation - 1,
                        step=1,
                        value=self.max_generation - 1,
                        marks={
                            i: str(i)
                            for i in range(
                                0,
                                self.max_generation,
                                max(1, self.max_generation // 10),
                            )
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Aggregated fitness plot (always visible)
            html.Div(
                [
                    html.H3("Aggregated Fitness Across Runs"),
                    html.P(
                        "Mean ± standard deviation across all runs",
                        style={"color": "#666", "fontSize": "14px"},
                    ),
                    dcc.Graph(id="aggregated-fitness"),
                ],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Individual runs plot (collapsible)
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Individual Run Trajectories",
                                style={"display": "inline-block", "margin": 0},
                            ),
                            html.Button(
                                "▼ Collapse",
                                id="runs-collapse-btn",
                                style={
                                    "float": "right",
                                    "background": "none",
                                    "border": "1px solid #ccc",
                                    "padding": "5px 10px",
                                    "cursor": "pointer",
                                    "borderRadius": "4px",
                                },
                            ),
                        ],
                        style={"marginBottom": "10px", "overflow": "hidden"},
                    ),
                    html.Div(
                        [dcc.Graph(id="individual-runs")],
                        id="runs-plot-container",
                        style={"display": "block"},
                    ),
                ],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Robot Viewer (collapsible)
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Robot Viewer",
                                style={"display": "inline-block", "margin": 0},
                            ),
                            html.Button(
                                "▶ Expand",
                                id="robot-viewer-collapse-btn",
                                style={
                                    "float": "right",
                                    "background": "none",
                                    "border": "1px solid #ccc",
                                    "padding": "5px 10px",
                                    "cursor": "pointer",
                                    "borderRadius": "4px",
                                },
                            ),
                        ],
                        style={"marginBottom": "10px", "overflow": "hidden"},
                    ),
                    html.Div(
                        [
                            # Target robot selector
                            html.Div(
                                [
                                    html.Label(
                                        "Select Target Robot:",
                                        style={
                                            "fontWeight": "bold",
                                            "marginRight": 10,
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="target-robot-dropdown",
                                        options=self._get_target_robot_options(),
                                        placeholder="Select a target robot to view...",
                                        style={
                                            "width": "300px",
                                            "display": "inline-block",
                                            "verticalAlign": "middle",
                                        },
                                    ),
                                ],
                                style={
                                    "textAlign": "center",
                                    "marginBottom": 20,
                                },
                            ),
                            # Direct robot selection
                            html.Div(
                                [
                                    html.Hr(style={"margin": "20px 0"}),
                                    html.Label(
                                        "Or Select Robot Directly:",
                                        style={
                                            "fontWeight": "bold",
                                            "marginBottom": 10,
                                            "display": "block",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            # Run selector
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Run:",
                                                        style={
                                                            "marginRight": 5
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="direct-run-dropdown",
                                                        options=[
                                                            {
                                                                "label": name,
                                                                "value": i,
                                                            }
                                                            for i, name in enumerate(
                                                                self.run_names
                                                            )
                                                        ],
                                                        value=0,
                                                        style={
                                                            "width": "150px"
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "inline-block",
                                                    "marginRight": 15,
                                                    "verticalAlign": "top",
                                                },
                                            ),
                                            # Generation selector
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Generation:",
                                                        style={
                                                            "marginRight": 5
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="direct-generation-dropdown",
                                                        options=[
                                                            {
                                                                "label": str(i),
                                                                "value": i,
                                                            }
                                                            for i in range(
                                                                self.max_generation
                                                            )
                                                        ],
                                                        placeholder="Gen...",
                                                        style={
                                                            "width": "100px"
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "inline-block",
                                                    "marginRight": 15,
                                                    "verticalAlign": "top",
                                                },
                                            ),
                                            # Individual selector (Best/Mean)
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Individual:",
                                                        style={
                                                            "marginRight": 5
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="direct-individual-dropdown",
                                                        options=[
                                                            {
                                                                "label": "Best",
                                                                "value": "best",
                                                            },
                                                            {
                                                                "label": "Mean",
                                                                "value": "mean",
                                                            },
                                                        ],
                                                        placeholder="Select...",
                                                        style={
                                                            "width": "120px"
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "inline-block",
                                                    "marginRight": 15,
                                                    "verticalAlign": "top",
                                                },
                                            ),
                                            # Load button
                                            html.Button(
                                                "Load Robot",
                                                id="direct-load-robot-btn",
                                                style={
                                                    "backgroundColor": "#4CAF50",
                                                    "color": "white",
                                                    "border": "none",
                                                    "padding": "8px 16px",
                                                    "cursor": "pointer",
                                                    "borderRadius": "4px",
                                                    "verticalAlign": "bottom",
                                                },
                                            ),
                                        ],
                                        style={"textAlign": "center"},
                                    ),
                                ],
                                style={"marginBottom": 20},
                            ),
                            html.Div(
                                id="robot-viewer-content",
                                children=[
                                    html.P(
                                        [
                                            "Click on any plot to view robots, select a target robot, or use direct selection above.",
                                            html.Br(),
                                            "The robot will be rendered using MuJoCo.",
                                        ],
                                        style={
                                            "textAlign": "center",
                                            "color": "#666",
                                            "fontSize": 14,
                                            "lineHeight": "1.8",
                                        },
                                    )
                                ],
                            ),
                        ],
                        id="robot-viewer-container",
                        style={"display": "none"},
                    ),
                ],
                style={
                    "margin": "20px",
                    "marginBottom": 40,
                    "padding": "20px",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "10px",
                    "border": "2px solid #ddd",
                },
            ),
            # Tabbed plots section
            html.Div(
                [
                    dcc.Tabs(
                        id="plot-tabs",
                        value="feature-evolution-tab",
                        children=[
                            dcc.Tab(
                                label="Feature Evolution",
                                value="feature-evolution-tab",
                            ),
                            dcc.Tab(
                                label="Fitness Distributions",
                                value="distribution-tab",
                            ),
                            dcc.Tab(
                                label="Diversity Metrics", value="diversity-tab"
                            ),
                            dcc.Tab(
                                label="Run Comparison", value="comparison-tab"
                            ),
                        ],
                    ),
                    html.Div(id="tab-content"),
                ],
                style={"margin": "20px"},
            ),
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [
                Output("runs-plot-container", "style"),
                Output("runs-collapse-btn", "children"),
            ],
            Input("runs-collapse-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_runs_plot(n_clicks):
            """Toggle individual runs plot visibility."""
            if n_clicks is None:
                n_clicks = 0

            if n_clicks % 2 == 0:
                return {"display": "block"}, "▼ Collapse"
            else:
                return {"display": "none"}, "▶ Expand"

        @self.app.callback(
            [
                Output("robot-viewer-container", "style"),
                Output("robot-viewer-collapse-btn", "children"),
            ],
            Input("robot-viewer-collapse-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_robot_viewer(n_clicks):
            """Toggle robot viewer visibility."""
            if n_clicks is None:
                n_clicks = 0

            if n_clicks % 2 == 0:
                # Hide viewer (starts collapsed)
                return {"display": "none"}, "▶ Expand"
            else:
                # Show viewer
                return {"display": "block"}, "▼ Collapse"

        @self.app.callback(
            Output("aggregated-fitness", "figure"),
            Input("generation-slider", "value"),
        )
        def update_aggregated_fitness(selected_generation):
            """Update aggregated fitness plot."""
            if not self.aggregated_timeline:
                return go.Figure()

            df = pd.DataFrame(self.aggregated_timeline)
            fig = go.Figure()

            # Add mean of average fitness across runs
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["mean_avg_fitness"],
                    mode="lines+markers",
                    name="Mean Avg Fitness",
                    line=dict(color="blue", width=3),
                    marker=dict(size=6),
                )
            )

            # Add std deviation band for average fitness
            fig.add_trace(
                go.Scatter(
                    x=df["generation"].tolist()
                    + df["generation"][::-1].tolist(),
                    y=(df["mean_avg_fitness"] + df["std_avg_fitness"]).tolist()
                    + (df["mean_avg_fitness"] - df["std_avg_fitness"])[
                        ::-1
                    ].tolist(),
                    fill="toself",
                    fillcolor="rgba(0,100,200,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=True,
                    name="±1 STD (Avg Fitness)",
                )
            )

            # Add mean of best fitness across runs
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["mean_best_fitness"],
                    mode="lines+markers",
                    name="Mean Best Fitness",
                    line=dict(color="green", width=3),
                    marker=dict(size=6),
                )
            )

            # Add std deviation band for best fitness
            fig.add_trace(
                go.Scatter(
                    x=df["generation"].tolist()
                    + df["generation"][::-1].tolist(),
                    y=(
                        df["mean_best_fitness"] + df["std_best_fitness"]
                    ).tolist()
                    + (df["mean_best_fitness"] - df["std_best_fitness"])[
                        ::-1
                    ].tolist(),
                    fill="toself",
                    fillcolor="rgba(0,200,100,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=True,
                    name="±1 STD (Best Fitness)",
                )
            )

            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(
                    go.Scatter(
                        x=[selected_row["generation"]],
                        y=[selected_row["mean_avg_fitness"]],
                        mode="markers",
                        name=f"Generation {selected_generation}",
                        marker=dict(
                            color="red",
                            size=15,
                            symbol="circle-open",
                            line=dict(width=3),
                        ),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title="Mean Fitness Evolution Across Runs",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=500,
                showlegend=True,
                hovermode="x unified",
            )

            return fig

        @self.app.callback(
            Output("individual-runs", "figure"),
            Input("generation-slider", "value"),
        )
        def update_individual_runs(selected_generation):
            """Update individual runs plot."""
            fig = go.Figure()

            for run_idx, timeline in enumerate(self.individual_run_timelines):
                if not timeline:
                    continue

                df = pd.DataFrame(timeline)
                color = self.run_colors[run_idx % len(self.run_colors)]

                # Add average fitness line for this run
                fig.add_trace(
                    go.Scatter(
                        x=df["generation"],
                        y=df["avg_fitness"],
                        mode="lines+markers",
                        name=f"{self.run_names[run_idx]} - Avg",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        opacity=0.7,
                    )
                )

                # Add best fitness line for this run
                fig.add_trace(
                    go.Scatter(
                        x=df["generation"],
                        y=df["best_fitness"],
                        mode="lines",
                        name=f"{self.run_names[run_idx]} - Best",
                        line=dict(color=color, width=1, dash="dot"),
                        showlegend=False,
                        opacity=0.5,
                    )
                )

            fig.update_layout(
                title="Individual Run Fitness Trajectories",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=500,
                showlegend=True,
                hovermode="x unified",
            )

            return fig

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("plot-tabs", "value"), Input("generation-slider", "value")],
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            if active_tab == "feature-evolution-tab":
                return self._create_feature_evolution_plot()
            elif active_tab == "distribution-tab":
                return self._create_distribution_plot(selected_generation)
            elif active_tab == "diversity-tab":
                return self._create_diversity_plot(selected_generation)
            elif active_tab == "comparison-tab":
                return self._create_comparison_plot(selected_generation)

            return html.Div("Select a tab to view plots")

        @self.app.callback(
            Output("feature-evolution-graph", "figure"),
            Input("feature-dropdown", "value"),
            prevent_initial_call=True,
        )
        def update_feature_plot(feature_idx):
            """Update feature evolution plot."""
            if feature_idx is None:
                return go.Figure()
            return self._plot_feature_evolution(feature_idx)

        @self.app.callback(
            [
                Output("status-message", "children"),
                Output("status-message", "style"),
            ],
            Input("comparison-scatter", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def handle_click(clickData, generation):
            """Handle click on scatter plot to save robot."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                point = clickData["points"][0]
                point_index = point.get(
                    "pointIndex", point.get("pointNumber", 0)
                )

                if (
                    not hasattr(self, "_click_data")
                    or point_index not in self._click_data
                ):
                    return "Error: Click data not found", {
                        "display": "block",
                        "backgroundColor": "#ffcccc",
                        "color": "red",
                        "textAlign": "center",
                        "marginBottom": 20,
                        "padding": "10px",
                        "borderRadius": "5px",
                    }

                click_info = self._click_data[point_index]
                run_idx = click_info["run_idx"]
                individual_idx = click_info["individual_idx"]

                cache_key = f"run{run_idx}_gen{generation}"
                if cache_key not in self._individual_cache:
                    return "Error: Individual data not found", {
                        "display": "block",
                        "backgroundColor": "#ffcccc",
                        "color": "red",
                        "textAlign": "center",
                        "marginBottom": 20,
                        "padding": "10px",
                        "borderRadius": "5px",
                    }

                individual_data = self._individual_cache[cache_key]
                population = individual_data["population"]
                decoder = individual_data["decoder"]

                if individual_idx >= len(population):
                    return f"Error: Individual index out of range", {
                        "display": "block",
                        "backgroundColor": "#ffcccc",
                        "color": "red",
                        "textAlign": "center",
                        "marginBottom": 20,
                        "padding": "10px",
                        "borderRadius": "5px",
                    }

                individual = population[individual_idx]
                robot_graph = decoder(individual)
                robot_data = json_graph.node_link_data(
                    robot_graph, edges="edges"
                )

                filename = f"robot_run{run_idx}_gen{generation}_ind{individual_idx}_fit{individual.fitness:.3f}.json"
                filepath = self.dl_robots_path / filename

                with open(filepath, "w") as f:
                    json.dump(robot_data, f, indent=2)

                # Render the robot
                rendered_img = self._render_robot(robot_graph)

                # Store the rendered robot for the viewer
                if rendered_img:
                    self.current_robot_image = {
                        "image": rendered_img,
                        "generation": generation,
                        "individual": individual_idx,
                        "fitness": individual.fitness,
                        "filename": filename,
                        "run_idx": run_idx,
                    }

                message = f"Robot saved: {filename}"
                if rendered_img:
                    message += " | Expand Robot Viewer to see it →"

                return message, {
                    "display": "block",
                    "backgroundColor": "#ccffcc",
                    "color": "green",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

            except Exception as e:
                return f"Error saving robot: {str(e)}", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

        # Callback to update robot viewer when scatter plot is clicked
        @self.app.callback(
            Output("robot-viewer-content", "children"),
            Input("comparison-scatter", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def update_robot_viewer(clickData, generation):
            """Update the robot viewer when a point is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Callback for target robot dropdown
        @self.app.callback(
            Output("robot-viewer-content", "children", allow_duplicate=True),
            Input("target-robot-dropdown", "value"),
            prevent_initial_call=True,
        )
        def display_target_robot(target_name):
            """Display selected target robot."""
            if not target_name or target_name not in self.target_robot_graphs:
                return dash.no_update

            robot_graph = self.target_robot_graphs[target_name]
            rendered_img = self._render_robot(robot_graph)

            if rendered_img:
                robot_info = {
                    "image": rendered_img,
                    "filename": f"{target_name}.json",
                    "is_target": True,
                    "name": target_name,
                }
                return self._render_target_robot_display(robot_info)

            return html.Div(
                "Failed to render target robot",
                style={"textAlign": "center", "color": "red"},
            )

        # Callback for direct robot selection
        @self.app.callback(
            [
                Output(
                    "robot-viewer-content", "children", allow_duplicate=True
                ),
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("direct-load-robot-btn", "n_clicks"),
            [
                State("direct-run-dropdown", "value"),
                State("direct-generation-dropdown", "value"),
                State("direct-individual-dropdown", "value"),
            ],
            prevent_initial_call=True,
        )
        def load_direct_robot(n_clicks, run_idx, generation, individual_type):
            """Load robot directly from dropdown selections."""
            if (
                not n_clicks
                or run_idx is None
                or generation is None
                or individual_type is None
            ):
                return dash.no_update, dash.no_update, dash.no_update

            try:
                populations, config = self.runs_data[run_idx]

                if generation >= len(populations):
                    generation = len(populations) - 1

                population = populations[generation]
                if not population:
                    return (
                        dash.no_update,
                        "No population data",
                        {
                            "display": "block",
                            "backgroundColor": "#ffcccc",
                            "color": "red",
                            "textAlign": "center",
                            "marginBottom": 20,
                            "padding": "10px",
                            "borderRadius": "5px",
                        },
                    )

                # Find the target individual
                if individual_type == "best":
                    individual = max(population, key=lambda x: x.fitness)
                    individual_idx = population.index(individual)
                else:  # mean
                    mean_fitness = sum(ind.fitness for ind in population) / len(
                        population
                    )
                    individual = min(
                        population, key=lambda x: abs(x.fitness - mean_fitness)
                    )
                    individual_idx = population.index(individual)

                # Decode and render
                decoder = lambda ind: config.genotype.from_json(
                    ind.genotype
                ).to_digraph()
                robot_graph = decoder(individual)

                # Save robot
                robot_data = json_graph.node_link_data(
                    robot_graph, edges="edges"
                )
                filename = "robot.json"
                filepath = self.dl_robots_path / filename

                with open(filepath, "w") as f:
                    json.dump(robot_data, f, indent=2)

                # Render the robot
                rendered_img = self._render_robot(robot_graph)

                if rendered_img:
                    self.current_robot_image = {
                        "image": rendered_img,
                        "generation": generation,
                        "individual": individual_idx,
                        "fitness": individual.fitness,
                        "filename": filename,
                        "run_idx": self.run_names[run_idx],
                    }

                    ind_str = "Best" if individual_type == "best" else "Mean"
                    message = f"{ind_str} robot from {self.run_names[run_idx]} Gen {generation} loaded"

                    return (
                        self._render_robot_display(self.current_robot_image),
                        message,
                        {
                            "display": "block",
                            "backgroundColor": "#ccffcc",
                            "color": "green",
                            "textAlign": "center",
                            "marginBottom": 20,
                            "padding": "10px",
                            "borderRadius": "5px",
                        },
                    )

                return (
                    dash.no_update,
                    "Failed to render robot",
                    {
                        "display": "block",
                        "backgroundColor": "#ffcccc",
                        "color": "red",
                        "textAlign": "center",
                        "marginBottom": 20,
                        "padding": "10px",
                        "borderRadius": "5px",
                    },
                )

            except Exception as e:
                return (
                    dash.no_update,
                    f"Error: {str(e)}",
                    {
                        "display": "block",
                        "backgroundColor": "#ffcccc",
                        "color": "red",
                        "textAlign": "center",
                        "marginBottom": 20,
                        "padding": "10px",
                        "borderRadius": "5px",
                    },
                )

        # Click handler for aggregated fitness plot
        @self.app.callback(
            [
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("aggregated-fitness", "clickData"),
            prevent_initial_call=True,
        )
        def handle_aggregated_fitness_click(clickData):
            """Handle clicks on aggregated fitness - renders best robot from clicked generation."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                point = clickData["points"][0]
                generation = int(point["x"])
                return self._handle_generation_click(generation, mode="best")
            except Exception as e:
                return f"Error: {str(e)}", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

        # Update robot viewer for aggregated fitness clicks
        @self.app.callback(
            Output("robot-viewer-content", "children", allow_duplicate=True),
            Input("aggregated-fitness", "clickData"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_aggregated(clickData):
            """Update robot viewer when aggregated fitness is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Click handler for individual runs plot
        @self.app.callback(
            [
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("individual-runs", "clickData"),
            prevent_initial_call=True,
        )
        def handle_individual_runs_click(clickData):
            """Handle clicks on individual runs - renders best robot from clicked generation."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                point = clickData["points"][0]
                generation = int(point["x"])
                return self._handle_generation_click(generation, mode="best")
            except Exception as e:
                return f"Error: {str(e)}", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

        # Update robot viewer for individual runs clicks
        @self.app.callback(
            Output("robot-viewer-content", "children", allow_duplicate=True),
            Input("individual-runs", "clickData"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_runs(clickData):
            """Update robot viewer when individual runs is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Click handler for feature evolution graph
        @self.app.callback(
            [
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("feature-evolution-graph", "clickData"),
            prevent_initial_call=True,
        )
        def handle_feature_evolution_click(clickData):
            """Handle clicks on feature evolution - renders robot closest to mean fitness."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                point = clickData["points"][0]
                generation = int(point["x"])
                return self._handle_generation_click(generation, mode="mean")
            except Exception as e:
                return f"Error: {str(e)}", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

        # Update robot viewer for feature evolution clicks
        @self.app.callback(
            Output("robot-viewer-content", "children", allow_duplicate=True),
            Input("feature-evolution-graph", "clickData"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_feature(clickData):
            """Update robot viewer when feature evolution is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

    def _handle_generation_click(self, generation: int, mode: str = "best"):
        """Handle click on a generation-based plot.

        Args:
            generation: The generation number clicked
            mode: 'best' for best fitness, 'mean' for closest to mean fitness

        Returns:
            Tuple of (message, style) for status display
        """
        try:
            # Use first run for simplicity
            populations, config = self.runs_data[0]

            if generation >= len(populations):
                generation = len(populations) - 1

            population = populations[generation]
            if not population:
                return "No population data", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                }

            # Find the target individual
            if mode == "best":
                individual = max(population, key=lambda x: x.fitness)
                individual_idx = population.index(individual)
            else:  # mode == 'mean'
                mean_fitness = sum(ind.fitness for ind in population) / len(
                    population
                )
                individual = min(
                    population, key=lambda x: abs(x.fitness - mean_fitness)
                )
                individual_idx = population.index(individual)

            # Decode and render
            decoder = lambda ind: config.genotype.from_json(
                ind.genotype
            ).to_digraph()
            robot_graph = decoder(individual)

            # Save robot
            robot_data = json_graph.node_link_data(robot_graph, edges="edges")
            filename = "robot.json"
            filepath = self.dl_robots_path / filename

            with open(filepath, "w") as f:
                json.dump(robot_data, f, indent=2)

            # Render the robot
            rendered_img = self._render_robot(robot_graph)

            if rendered_img:
                self.current_robot_image = {
                    "image": rendered_img,
                    "generation": generation,
                    "individual": individual_idx,
                    "fitness": individual.fitness,
                    "filename": filename,
                    "run_idx": 0,
                }

            mode_str = "Best" if mode == "best" else "Mean"
            message = f"{mode_str} robot from Gen {generation} saved | Expand Robot Viewer →"

            return message, {
                "display": "block",
                "backgroundColor": "#ccffcc",
                "color": "green",
                "textAlign": "center",
                "marginBottom": 20,
                "padding": "10px",
                "borderRadius": "5px",
            }

        except Exception as e:
            return f"Error: {str(e)}", {
                "display": "block",
                "backgroundColor": "#ffcccc",
                "color": "red",
                "textAlign": "center",
                "marginBottom": 20,
                "padding": "10px",
                "borderRadius": "5px",
            }

    def _render_robot(self, robot_graph, width=800, height=600):
        """Render a robot using MuJoCo and return base64-encoded image.

        Args:
            robot_graph: NetworkX DiGraph representing the robot
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Base64-encoded PNG image string, or None if rendering fails
        """
        try:
            # Disable MuJoCo controller callback
            mj.set_mjcb_control(None)

            # Construct robot MuJoCo spec from graph
            robot_spec = construct_mjspec_from_graph(robot_graph)

            # Create world and spawn robot at origin
            world = OlympicArena()
            spawn_pos = [
                0,
                0,
                0.1,
            ]  # Spawn at world origin, slightly above ground
            world.spawn(robot_spec.spec, spawn_position=spawn_pos)

            # Compile model and create data
            model = world.spec.compile()
            data = mj.MjData(model)

            # Create camera positioned to view the whole robot
            camera = mj.MjvCamera()
            camera.type = mj.mjtCamera.mjCAMERA_FREE
            camera.lookat = [0, 0, 0.3]  # Look at slightly above ground
            camera.distance = 2.5  # Distance from robot to see it fully
            camera.azimuth = 45  # Angle around vertical axis
            camera.elevation = -20  # Angle above horizontal

            # Render single frame
            img = single_frame_renderer(
                model,
                data,
                steps=1,
                save=False,
                show=False,
                camera=camera,
                width=width,
                height=height,
            )

            # Convert PIL Image to base64 string
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return img_str

        except Exception as e:
            print(f"Error rendering robot: {str(e)}")
            return None

    def _render_robot_display(self, robot_info):
        """Render robot display HTML for the robot viewer.

        Args:
            robot_info: Dictionary with robot information

        Returns:
            html.Div containing robot display
        """
        return html.Div([
            # Robot metadata
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong("Run: "),
                            html.Span(f"{robot_info.get('run_idx', 'N/A')}"),
                        ],
                        style={"display": "inline-block", "marginRight": 30},
                    ),
                    html.Div(
                        [
                            html.Strong("Generation: "),
                            html.Span(f"{robot_info['generation']}"),
                        ],
                        style={"display": "inline-block", "marginRight": 30},
                    ),
                    html.Div(
                        [
                            html.Strong("Individual: "),
                            html.Span(f"{robot_info['individual']}"),
                        ],
                        style={"display": "inline-block", "marginRight": 30},
                    ),
                    html.Div(
                        [
                            html.Strong("Fitness: "),
                            html.Span(f"{robot_info['fitness']:.4f}"),
                        ],
                        style={"display": "inline-block"},
                    ),
                ],
                style={
                    "textAlign": "center",
                    "marginBottom": 20,
                    "fontSize": 14,
                },
            ),
            # Rendered robot image
            html.Div(
                [
                    html.Img(
                        src=f"data:image/png;base64,{robot_info['image']}",
                        style={
                            "maxWidth": "100%",
                            "height": "auto",
                            "border": "2px solid #ddd",
                            "borderRadius": "8px",
                            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        },
                    )
                ],
                style={"textAlign": "center", "padding": "20px"},
            ),
            # Additional info
            html.Div([
                html.P(
                    f"Saved as: {robot_info['filename']}",
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "fontSize": 12,
                    },
                ),
                html.P(
                    "Camera view: 45° azimuth, -20° elevation, 2.5m distance",
                    style={
                        "textAlign": "center",
                        "color": "#999",
                        "fontSize": 11,
                    },
                ),
            ]),
        ])

    def _render_target_robot_display(self, robot_info):
        """Render target robot display HTML for the robot viewer.

        Args:
            robot_info: Dictionary with target robot information

        Returns:
            html.Div containing target robot display
        """
        return html.Div([
            # Target robot header
            html.Div(
                [
                    html.Strong("Target Robot: ", style={"fontSize": 16}),
                    html.Span(
                        f"{robot_info['name']}",
                        style={"fontSize": 16, "color": "#2e7d32"},
                    ),
                ],
                style={"textAlign": "center", "marginBottom": 20},
            ),
            # Rendered robot image
            html.Div(
                [
                    html.Img(
                        src=f"data:image/png;base64,{robot_info['image']}",
                        style={
                            "maxWidth": "100%",
                            "height": "auto",
                            "border": "3px solid #2e7d32",
                            "borderRadius": "8px",
                            "boxShadow": "0 4px 6px rgba(46,125,50,0.3)",
                        },
                    )
                ],
                style={"textAlign": "center", "padding": "20px"},
            ),
            # Additional info
            html.Div([
                html.P(
                    f"File: {robot_info['filename']}",
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "fontSize": 12,
                    },
                )
            ]),
        ])

    def _create_feature_evolution_plot(self):
        """Create feature evolution plot with feature selector."""
        return html.Div([
            html.Div(
                [
                    html.Label(
                        "Select Morphological Feature:",
                        style={"fontWeight": "bold", "marginBottom": 10},
                    ),
                    dcc.Dropdown(
                        id="feature-dropdown",
                        options=[
                            {"label": feature, "value": i}
                            for i, feature in enumerate(self.feature_names)
                        ],
                        value=0,
                        style={"marginBottom": 20},
                    ),
                ],
                style={"width": "300px", "margin": "0 auto"},
            ),
            dcc.Graph(id="feature-evolution-graph"),
        ])

    def _plot_feature_evolution(self, feature_idx: int):
        """Plot evolution of a single morphological feature across runs."""
        if feature_idx is None:
            return go.Figure()

        feature_name = self.feature_names[feature_idx]

        # Compute aggregated feature values across runs
        aggregated_data = []

        for gen_idx in range(self.max_generation):
            # Collect feature statistics from all runs at this generation
            feature_means = []
            feature_stds = []

            for run_idx in range(self.num_runs):
                populations, _ = self.runs_data[run_idx]
                if gen_idx < len(populations):
                    gen_data = self._get_generation_data(run_idx, gen_idx)
                    if gen_data["descriptors"].size > 0:
                        feature_values = gen_data["descriptors"][:, feature_idx]
                        feature_means.append(np.mean(feature_values))
                        feature_stds.append(np.std(feature_values))

            if feature_means:
                aggregated_data.append({
                    "generation": gen_idx,
                    "mean_of_means": np.mean(feature_means),
                    "std_of_means": np.std(feature_means),
                    "mean_diversity": np.mean(feature_stds),
                })

        if not aggregated_data:
            return go.Figure()

        df = pd.DataFrame(aggregated_data)
        fig = go.Figure()

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["mean_of_means"],
                mode="lines+markers",
                name="Mean Across Runs",
                line=dict(color="blue", width=3),
                marker=dict(size=6),
            )
        )

        # Add std deviation band
        fig.add_trace(
            go.Scatter(
                x=df["generation"].tolist() + df["generation"][::-1].tolist(),
                y=(df["mean_of_means"] + df["std_of_means"]).tolist()
                + (df["mean_of_means"] - df["std_of_means"])[::-1].tolist(),
                fill="toself",
                fillcolor="rgba(0,100,200,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="±1 STD Across Runs",
            )
        )

        # Add mean within-run diversity
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["mean_diversity"],
                mode="lines+markers",
                name="Mean Within-Run Diversity",
                line=dict(color="purple", width=2, dash="dot"),
                marker=dict(size=5),
            )
        )

        # Add target robot values if available
        if (
            hasattr(self.analyzer, "target_descriptors")
            and len(self.analyzer.target_descriptors) > 0
        ):
            for i, (target_desc, target_name) in enumerate(
                zip(
                    self.analyzer.target_descriptors, self.analyzer.target_names
                )
            ):
                target_value = target_desc[feature_idx]
                fig.add_hline(
                    y=target_value,
                    line_dash="solid",
                    line_color=px.colors.qualitative.Set1[
                        i % len(px.colors.qualitative.Set1)
                    ],
                    line_width=2,
                    annotation_text=f"Target: {target_name} ({target_value:.2f})",
                    annotation_position="top right",
                )

        fig.update_layout(
            title=f"{feature_name} Evolution Across Multiple Runs",
            xaxis_title="Generation",
            yaxis_title=f"{feature_name} Value",
            height=500,
            showlegend=True,
            hovermode="x unified",
        )

        return fig

    def _create_distribution_plot(self, generation: int):
        """Create fitness distribution comparison across runs."""
        fig = go.Figure()

        for run_idx in range(self.num_runs):
            populations, _ = self.runs_data[run_idx]

            if generation >= len(populations):
                continue

            population = populations[generation]
            if not population:
                continue

            fitnesses = [ind.fitness for ind in population]
            color = self.run_colors[run_idx % len(self.run_colors)]

            fig.add_trace(
                go.Histogram(
                    x=fitnesses,
                    name=self.run_names[run_idx],
                    opacity=0.6,
                    nbinsx=20,
                    marker_color=color,
                )
            )

        fig.update_layout(
            title=f"Fitness Distributions Across Runs - Generation {generation}",
            xaxis_title="Fitness",
            yaxis_title="Count",
            height=500,
            barmode="overlay",
            showlegend=True,
        )

        return dcc.Graph(figure=fig)

    def _create_diversity_plot(self, generation: int):
        """Create morphological diversity comparison across runs."""
        fig = go.Figure()

        for run_idx in range(self.num_runs):
            gen_data = self._get_generation_data(run_idx, generation)

            if gen_data["descriptors"].size == 0:
                continue

            descriptors = gen_data["descriptors"]
            color = self.run_colors[run_idx % len(self.run_colors)]

            # Compute diversity (std) for each feature
            diversity_data = []
            for i in range(len(self.feature_names)):
                feature_values = descriptors[:, i]
                diversity = np.std(feature_values)
                diversity_data.append(diversity)

            fig.add_trace(
                go.Scatterpolar(
                    r=diversity_data,
                    theta=self.feature_names,
                    fill="toself",
                    name=self.run_names[run_idx],
                    line_color=color,
                    fillcolor=self._color_to_rgba(color, alpha=0.3),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, None])),
            title=f"Morphological Diversity Across Runs - Generation {generation}",
            height=600,
            showlegend=True,
        )

        return dcc.Graph(figure=fig)

    def _create_comparison_plot(self, generation: int):
        """Create scatter plot comparing individuals from all runs."""
        fig = go.Figure()

        # Store mapping for click handling
        self._click_data = {}
        current_index = 0

        for run_idx in range(self.num_runs):
            gen_data = self._get_generation_data(run_idx, generation)

            if gen_data["descriptors"].size == 0:
                continue

            descriptors = gen_data["descriptors"]
            fitness_scores = gen_data["fitness_scores"]
            color = self.run_colors[run_idx % len(self.run_colors)]

            # Use first two features for scatter
            x_values = descriptors[:, 0]  # Branching
            y_values = descriptors[:, 1]  # Limbs

            # Store click data mapping
            for i in range(len(x_values)):
                self._click_data[current_index] = {
                    "run_idx": run_idx,
                    "individual_idx": i,
                    "generation": generation,
                }
                current_index += 1

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name=self.run_names[run_idx],
                    marker=dict(
                        color=fitness_scores[0]
                        if len(fitness_scores) > 0
                        else [0] * len(x_values),
                        colorscale="Viridis",
                        size=8,
                        opacity=0.6,
                        line=dict(width=0.5, color="white"),
                        colorbar=dict(title="Fitness")
                        if run_idx == 0
                        else None,
                    ),
                    hovertemplate=f"<b>{self.run_names[run_idx]}</b><br>"
                    + "Branching: %{x:.3f}<br>"
                    + "Limbs: %{y:.3f}<br>"
                    + "Fitness: %{marker.color:.3f}<br>"
                    + "<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Morphological Space Comparison - Generation {generation}<br><sub>Click on any point to download robot</sub>",
            xaxis_title="Branching",
            yaxis_title="Limbs",
            height=600,
            showlegend=True,
        )

        return dcc.Graph(id="comparison-scatter", figure=fig)

    def run(self, host="127.0.0.1", port=8052, debug=True):
        """Run the dashboard server."""
        print(f"Starting Multiple Runs Dashboard at http://{host}:{port}")
        print(f"Analyzing {self.num_runs} independent runs")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def load_config_from_saved_json(config_path: Path) -> Any:
    """Load configuration from saved JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
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
            self.task_params = resolved["task_params"]
            self.genotype_name = genotype_name

    return DashboardConfig()


def load_populations_from_database(
    db_path: Path,
) -> Tuple[List[Population], Any]:
    """Load populations from a single database file."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    console.log(f"Loading data from: {db_path}")

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

            population = [Individual(**ind.model_dump()) for ind in individuals]
            populations.append(population)

    # Load config
    config_filename = db_path.stem + "_config.json"
    json_config_path = db_path.parent / config_filename

    if json_config_path.exists():
        console.log(f"Loading configuration from: {json_config_path}")
        config = load_config_from_saved_json(json_config_path)
    else:
        raise FileNotFoundError(
            f"No configuration file found for database {db_path}"
        )

    return populations, config


def main():
    """Main entry point for multiple runs dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multiple runs analysis dashboard"
    )
    parser.add_argument(
        "--db_paths",
        nargs="+",
        default=[
            "__data__/multiple_runs/run0.db",
            "__data__/multiple_runs/run1.db",
            "__data__/multiple_runs/run2.db",
            "__data__/multiple_runs/run3.db",
            "__data__/multiple_runs/run4.db",
        ],
        help="Paths to database files (one per run)",
    )
    parser.add_argument(
        "--names", nargs="+", help="Names for runs (default: Run 1, Run 2, ...)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8052, help="Dashboard port")
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable debug mode"
    )

    args = parser.parse_args()

    # Load all runs
    runs_data = []
    for i, db_path in enumerate(args.db_paths):
        db_path = Path(db_path)
        populations, config = load_populations_from_database(db_path)
        runs_data.append((populations, config))
        console.log(f"Loaded run {i + 1}: {len(populations)} generations")

    if not runs_data:
        raise ValueError("No runs loaded!")

    # Set run names
    if args.names and len(args.names) == len(runs_data):
        run_names = args.names
    else:
        run_names = [f"Run {i + 1}" for i in range(len(runs_data))]

    console.log(f"Starting dashboard with {len(runs_data)} runs")
    dashboard = MultipleRunsDashboard(runs_data, run_names)
    dashboard.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()
