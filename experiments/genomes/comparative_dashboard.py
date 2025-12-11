#!/usr/bin/env python3
"""
Comparative Evolution Dashboard

This module provides an interactive web dashboard for comparing evolutionary
computation results across different genotypes using Plotly Dash. It loads
data from multiple SQLite databases and displays comparative analysis.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
import tomllib
import json
import os
import base64
from io import BytesIO
from typing import List, Callable, Any, Dict, Tuple
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


class ComparativeEvolutionDashboard:
    """Interactive dashboard for comparing evolution across genotypes."""

    def __init__(self, genotype_data: Dict[str, Tuple[List[Population], Any]]):
        """Initialize dashboard with multi-genotype evolution data.

        Args:
            genotype_data: Dict mapping genotype names to (populations, config) tuples
        """
        self.genotype_data = genotype_data
        self.genotype_names = list(genotype_data.keys())
        self.analyzers = {}

        # Initialize analyzer for each genotype
        for genotype_name, (populations, config) in genotype_data.items():
            analyzer = PlotlyMorphologyAnalyzer()
            # Load target robots if available
            if (
                hasattr(config, "task_params")
                and "target_robot_path" in config.task_params
            ):
                analyzer.load_target_robots(
                    str(config.task_params["target_robot_path"])
                )
            self.analyzers[genotype_name] = analyzer

        # Pre-compute fitness timelines for all genotypes
        self._compute_fitness_timelines()

        # Compute max generation across all genotypes
        self.max_generation = max(
            len(populations) for populations, _ in self.genotype_data.values()
        )

        # Cache for computed descriptors per generation per genotype
        self._descriptor_cache = {}

        # Cache for individual data per generation per genotype (for click handling)
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

        # Color mapping for genotypes
        self.colors = {
            self.genotype_names[0]: "#1f77b4",  # Blue
            self.genotype_names[1]: "#ff7f0e"
            if len(self.genotype_names) > 1
            else "#1f77b4",  # Orange
            self.genotype_names[2]
            if len(self.genotype_names) > 2
            else "": "#2ca02c"
            if len(self.genotype_names) > 2
            else "#1f77b4",  # Green
        }

        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()

    def _compute_fitness_timelines(self):
        """Compute average fitness for each generation for each genotype."""
        self.fitness_timelines = {}

        for genotype_name, (populations, config) in self.genotype_data.items():
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

            self.fitness_timelines[genotype_name] = timeline

    def _load_target_robot_graphs(self):
        """Load target robot graphs from JSON files for all genotypes."""
        self.target_robot_graphs = {}

        for genotype_name, (populations, config) in self.genotype_data.items():
            if (
                hasattr(config, "task_params")
                and "target_robot_path" in config.task_params
            ):
                target_path = Path(config.task_params["target_robot_path"])

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
                            # Avoid duplicates
                            if robot_name not in self.target_robot_graphs:
                                self.target_robot_graphs[robot_name] = (
                                    robot_graph
                                )
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

    def _get_generation_data(self, genotype_name: str, generation: int):
        """Get or compute morphological data for a specific genotype and generation."""
        cache_key = f"{genotype_name}_{generation}"

        if cache_key in self._descriptor_cache:
            return self._descriptor_cache[cache_key]

        populations, config = self.genotype_data[genotype_name]

        if generation >= len(populations):
            generation = len(populations) - 1

        # Load population data into analyzer
        population = populations[generation]
        analyzer = self.analyzers[genotype_name]

        def decoder(individual: Individual):
            return config.genotype.from_json(individual.genotype).to_digraph()

        analyzer.load_population(population, decoder)
        analyzer.compute_fitness_scores()

        # Cache the results
        self._descriptor_cache[cache_key] = {
            "descriptors": analyzer.descriptors.copy(),
            "fitness_scores": analyzer.fitness_scores.copy(),
        }

        # Cache individual data for click handling
        individual_cache_key = f"{genotype_name}_{generation}"
        self._individual_cache[individual_cache_key] = {
            "population": population.copy(),
            "decoder": decoder,
            "config": config,
        }

        return self._descriptor_cache[cache_key]

    def _setup_layout(self):
        """Setup the dashboard layout."""
        # Find max generation across all genotypes
        max_generation = max(
            len(populations) - 1
            for populations, _ in self.genotype_data.values()
        )

        self.app.layout = html.Div([
            html.H1(
                "Comparative Evolution Dashboard",
                style={"textAlign": "center", "marginBottom": 30},
            ),
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
                        max=max_generation,
                        step=1,
                        value=max_generation,
                        marks={
                            i: str(i)
                            for i in range(
                                0,
                                max_generation + 1,
                                max(1, max_generation // 10),
                            )
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Fitness comparison (collapsible)
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Fitness Comparison Across Genotypes",
                                style={"display": "inline-block", "margin": 0},
                            ),
                            html.Button(
                                "▼ Collapse",
                                id="fitness-collapse-btn",
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
                        [dcc.Graph(id="fitness-comparison")],
                        id="fitness-plot-container",
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
                                            # Genotype selector
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Genotype:",
                                                        style={
                                                            "marginRight": 5
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="direct-genotype-dropdown",
                                                        options=[
                                                            {
                                                                "label": name,
                                                                "value": name,
                                                            }
                                                            for name in self.genotype_names
                                                        ],
                                                        value=self.genotype_names[
                                                            0
                                                        ]
                                                        if self.genotype_names
                                                        else None,
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
                                            # Individual selector (Best/Mean/Individual index)
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
                        value="single-feature-tab",
                        children=[
                            dcc.Tab(
                                label="Single Feature Evolution",
                                value="single-feature-tab",
                            ),
                            dcc.Tab(
                                label="Fitness Distributions",
                                value="distribution-tab",
                            ),
                            dcc.Tab(
                                label="Morphological Diversity",
                                value="diversity-tab",
                            ),
                            dcc.Tab(
                                label="Individual Analysis",
                                value="individual-tab",
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
                Output("fitness-plot-container", "style"),
                Output("fitness-collapse-btn", "children"),
            ],
            Input("fitness-collapse-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_fitness_plot(n_clicks):
            """Toggle fitness plot visibility."""
            if n_clicks is None:
                n_clicks = 0

            if n_clicks % 2 == 0:
                # Show plot
                return {"display": "block"}, "▼ Collapse"
            else:
                # Hide plot
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
            Output("fitness-comparison", "figure"),
            Input("generation-slider", "value"),
        )
        def update_fitness_comparison(selected_generation):
            """Update fitness comparison plot with highlighted generation."""
            fig = go.Figure()

            for genotype_name in self.genotype_names:
                timeline = self.fitness_timelines[genotype_name]
                if not timeline:
                    continue

                df = pd.DataFrame(timeline)
                color = self.colors.get(genotype_name, "#1f77b4")

                # Add best fitness line
                fig.add_trace(
                    go.Scatter(
                        x=df["generation"],
                        y=df["best_fitness"],
                        mode="lines+markers",
                        name=f"{genotype_name} - Best",
                        line=dict(color=color, width=2, dash="solid"),
                        marker=dict(size=5, symbol="star"),
                    )
                )

                # Add mean fitness line
                fig.add_trace(
                    go.Scatter(
                        x=df["generation"],
                        y=df["avg_fitness"],
                        mode="lines+markers",
                        name=f"{genotype_name} - Mean",
                        line=dict(color=color, width=2, dash="dash"),
                        marker=dict(size=4),
                    )
                )

                # Add std deviation band
                fig.add_trace(
                    go.Scatter(
                        x=df["generation"].tolist()
                        + df["generation"][::-1].tolist(),
                        y=(df["avg_fitness"] + df["std_fitness"]).tolist()
                        + (df["avg_fitness"] - df["std_fitness"])[
                            ::-1
                        ].tolist(),
                        fill="toself",
                        fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.15])}",
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        name=f"{genotype_name} - ±1 STD",
                    )
                )

                # Highlight selected generation
                if selected_generation < len(df):
                    selected_row = df.iloc[selected_generation]
                    # Highlight best
                    fig.add_trace(
                        go.Scatter(
                            x=[selected_row["generation"]],
                            y=[selected_row["best_fitness"]],
                            mode="markers",
                            name=f"{genotype_name} - Best Gen {selected_generation}",
                            marker=dict(
                                color=color,
                                size=14,
                                symbol="star-open",
                                line=dict(width=3),
                            ),
                            showlegend=False,
                        )
                    )
                    # Highlight mean
                    fig.add_trace(
                        go.Scatter(
                            x=[selected_row["generation"]],
                            y=[selected_row["avg_fitness"]],
                            mode="markers",
                            name=f"{genotype_name} - Mean Gen {selected_generation}",
                            marker=dict(
                                color=color,
                                size=12,
                                symbol="circle-open",
                                line=dict(width=3),
                            ),
                            showlegend=False,
                        )
                    )

            fig.update_layout(
                title="Fitness Evolution: Best and Mean with Standard Deviation",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=500,
                showlegend=True,
                hovermode="x unified",
            )

            return fig

        @self.app.callback(
            Output("single-feature-graph", "figure"),
            Input("feature-dropdown", "value"),
            prevent_initial_call=True,
        )
        def update_single_feature_plot(feature_idx):
            """Update single feature plot based on selected feature."""
            if feature_idx is None:
                return go.Figure()
            return self._plot_feature_evolution_comparison(feature_idx)

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("plot-tabs", "value"), Input("generation-slider", "value")],
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            if active_tab == "distribution-tab":
                return self._create_distribution_comparison(selected_generation)
            elif active_tab == "diversity-tab":
                return self._create_diversity_comparison(selected_generation)
            elif active_tab == "individual-tab":
                return self._create_individual_analysis(selected_generation)
            elif active_tab == "single-feature-tab":
                return self._create_single_feature_plot()

            return html.Div("Select a tab to view plots")

        # Direct click handling - save robot immediately when clicked
        @self.app.callback(
            [
                Output("status-message", "children"),
                Output("status-message", "style"),
            ],
            Input("individual-scatter", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def handle_direct_click(clickData, generation):
            """Handle direct click on scatter plot and save robot immediately."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                # Get the clicked point index
                point = clickData["points"][0]
                point_index = point.get(
                    "pointIndex", point.get("pointNumber", 0)
                )

                # Get the individual from click data mapping
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
                genotype_name = click_info["genotype"]
                individual_index = click_info["individual_index"]

                # Get the individual data
                cache_key = f"{genotype_name}_{generation}"
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

                if individual_index >= len(population):
                    return (
                        f"Error: Individual index {individual_index} out of range",
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

                # Get the individual and convert to robot graph
                individual = population[individual_index]
                robot_graph = decoder(individual)

                # Save robot as JSON
                robot_data = json_graph.node_link_data(
                    robot_graph, edges="edges"
                )

                # Save to file
                filename = "robot.json"
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
                        "individual": individual_index,
                        "fitness": individual.fitness,
                        "filename": filename,
                        "genotype": genotype_name,
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
            Input("individual-scatter", "clickData"),
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
                State("direct-genotype-dropdown", "value"),
                State("direct-generation-dropdown", "value"),
                State("direct-individual-dropdown", "value"),
            ],
            prevent_initial_call=True,
        )
        def load_direct_robot(
            n_clicks, genotype_name, generation, individual_type
        ):
            """Load robot directly from dropdown selections."""
            if (
                not n_clicks
                or genotype_name is None
                or generation is None
                or individual_type is None
            ):
                return dash.no_update, dash.no_update, dash.no_update

            try:
                populations, config = self.genotype_data[genotype_name]

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
                        "genotype": genotype_name,
                    }

                    ind_str = "Best" if individual_type == "best" else "Mean"
                    message = f"{ind_str} robot from {genotype_name} Gen {generation} loaded"

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

        # Click handler for fitness comparison plot
        @self.app.callback(
            [
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("fitness-comparison", "clickData"),
            prevent_initial_call=True,
        )
        def handle_fitness_comparison_click(clickData):
            """Handle clicks on fitness comparison - renders best robot from clicked generation."""
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

        # Update robot viewer for fitness comparison clicks
        @self.app.callback(
            Output("robot-viewer-content", "children", allow_duplicate=True),
            Input("fitness-comparison", "clickData"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_fitness(clickData):
            """Update robot viewer when fitness comparison is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Click handler for single feature graph
        @self.app.callback(
            [
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("single-feature-graph", "clickData"),
            prevent_initial_call=True,
        )
        def handle_single_feature_click(clickData):
            """Handle clicks on single feature graph - renders robot closest to mean fitness."""
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

        # Update robot viewer for single feature clicks
        @self.app.callback(
            Output("robot-viewer-content", "children", allow_duplicate=True),
            Input("single-feature-graph", "clickData"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_feature(clickData):
            """Update robot viewer when single feature graph is clicked."""
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
            # Use first genotype for simplicity
            genotype_name = self.genotype_names[0]
            populations, config = self.genotype_data[genotype_name]

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
                    "genotype": genotype_name,
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
                            html.Strong("Genotype: "),
                            html.Span(f"{robot_info.get('genotype', 'N/A')}"),
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

    def _create_single_feature_plot(self):
        """Create single feature evolution comparison with feature selector."""
        return html.Div([
            html.Div(
                [
                    html.Label(
                        "Select Feature:",
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
            dcc.Graph(id="single-feature-graph"),
        ])

    def _plot_feature_evolution_comparison(self, feature_idx: int):
        """Plot evolution of a single morphological feature across genotypes."""
        if feature_idx is None:
            return go.Figure()

        feature_name = self.feature_names[feature_idx]
        fig = go.Figure()

        for genotype_name in self.genotype_names:
            populations, config = self.genotype_data[genotype_name]
            color = self.colors.get(genotype_name, "#1f77b4")

            # Compute feature values for all generations
            generation_data = []

            for gen_idx in range(len(populations)):
                gen_data = self._get_generation_data(genotype_name, gen_idx)
                if gen_data["descriptors"].size > 0:
                    feature_values = gen_data["descriptors"][:, feature_idx]

                    generation_data.append({
                        "generation": gen_idx,
                        "mean": np.mean(feature_values),
                        "std": np.std(feature_values),
                    })

            if not generation_data:
                continue

            df = pd.DataFrame(generation_data)

            # Add mean line
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["mean"],
                    mode="lines+markers",
                    name=f"{genotype_name} - Mean",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                )
            )

            # Add standard deviation bands
            fig.add_trace(
                go.Scatter(
                    x=df["generation"].tolist()
                    + df["generation"][::-1].tolist(),
                    y=(df["mean"] + df["std"]).tolist()
                    + (df["mean"] - df["std"])[::-1].tolist(),
                    fill="tonext",
                    fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    name=f"{genotype_name} - ±1 STD",
                )
            )

        # Add target robot values as horizontal lines
        target_added = False
        for genotype_name in self.genotype_names:
            analyzer = self.analyzers[genotype_name]
            if (
                hasattr(analyzer, "target_descriptors")
                and len(analyzer.target_descriptors) > 0
            ):
                # Since there's only one target robot per genotype, use the first one
                target_value = analyzer.target_descriptors[0, feature_idx]
                target_name = (
                    analyzer.target_names[0]
                    if analyzer.target_names
                    else "Target"
                )

                # Only add one target line (they should be the same across genotypes)
                if not target_added:
                    fig.add_hline(
                        y=target_value,
                        line_dash="solid",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"Target: {target_name} ({target_value:.2f})",
                        annotation_position="top right",
                    )
                    target_added = True
                break  # Only need one target since they should be the same

        fig.update_layout(
            title=f"{feature_name} Evolution Comparison",
            xaxis_title="Generation",
            yaxis_title=f"{feature_name} Value",
            height=500,
            showlegend=True,
            hovermode="x unified",
        )

        return fig

    def _create_distribution_comparison(self, generation: int):
        """Create fitness distribution comparison for selected generation."""
        fig = go.Figure()

        for genotype_name in self.genotype_names:
            populations, config = self.genotype_data[genotype_name]

            if generation >= len(populations):
                continue

            population = populations[generation]
            if not population:
                continue

            fitnesses = [ind.fitness for ind in population]
            color = self.colors.get(genotype_name, "#1f77b4")

            fig.add_trace(
                go.Histogram(
                    x=fitnesses,
                    name=genotype_name,
                    opacity=0.7,
                    nbinsx=20,
                    marker_color=color,
                )
            )

        fig.update_layout(
            title=f"Fitness Distributions - Generation {generation}",
            xaxis_title="Fitness",
            yaxis_title="Count",
            height=500,
            barmode="overlay",
            showlegend=True,
        )

        return dcc.Graph(figure=fig)

    def _create_diversity_comparison(self, generation: int):
        """Create morphological diversity comparison for selected generation."""
        fig = go.Figure()

        for genotype_name in self.genotype_names:
            gen_data = self._get_generation_data(genotype_name, generation)

            if gen_data["descriptors"].size == 0:
                continue

            descriptors = gen_data["descriptors"]
            color = self.colors.get(genotype_name, "#1f77b4")

            # Compute diversity metrics for each feature
            diversity_data = []
            for i, feature_name in enumerate(self.feature_names):
                feature_values = descriptors[:, i]
                diversity = np.std(feature_values)  # Simple diversity measure
                diversity_data.append(diversity)

            fig.add_trace(
                go.Scatterpolar(
                    r=diversity_data,
                    theta=self.feature_names,
                    fill="toself",
                    name=genotype_name,
                    line_color=color,
                    fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.3])}",
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[
                        0,
                        max([
                            max(
                                self._get_diversity_for_genotype(
                                    gname, generation
                                )
                            )
                            for gname in self.genotype_names
                            if self._get_diversity_for_genotype(
                                gname, generation
                            )
                        ])
                        if any(
                            self._get_diversity_for_genotype(gname, generation)
                            for gname in self.genotype_names
                        )
                        else 1,
                    ],
                )
            ),
            title=f"Morphological Diversity Comparison - Generation {generation}",
            height=500,
            showlegend=True,
        )

        return dcc.Graph(figure=fig)

    def _get_diversity_for_genotype(
        self, genotype_name: str, generation: int
    ) -> List[float]:
        """Helper to get diversity values for a genotype at a generation."""
        try:
            gen_data = self._get_generation_data(genotype_name, generation)
            if gen_data["descriptors"].size == 0:
                return []

            descriptors = gen_data["descriptors"]
            return [
                np.std(descriptors[:, i])
                for i in range(len(self.feature_names))
            ]
        except:
            return []

    def _create_individual_analysis(self, generation: int):
        """Create individual analysis scatter plot for selected generation."""
        fig = go.Figure()

        # Store individual indices for click handling
        self._click_data = {}
        current_index = 0

        for genotype_name in self.genotype_names:
            gen_data = self._get_generation_data(genotype_name, generation)

            if gen_data["descriptors"].size == 0:
                continue

            descriptors = gen_data["descriptors"]
            fitness_scores = gen_data["fitness_scores"]
            color = self.colors.get(genotype_name, "#1f77b4")

            # Use first two morphological features for scatter plot
            x_values = descriptors[:, 0]  # Branching
            y_values = descriptors[:, 1]  # Limbs

            # Store mapping from plot index to genotype and individual index
            for i in range(len(x_values)):
                self._click_data[current_index] = {
                    "genotype": genotype_name,
                    "individual_index": i,
                    "generation": generation,
                }
                current_index += 1

            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name=genotype_name,
                    marker=dict(
                        color=fitness_scores[0]
                        if len(fitness_scores) > 0
                        else [0] * len(x_values),
                        colorscale="Viridis",
                        size=8,
                        opacity=0.7,
                        colorbar=dict(title="Fitness")
                        if genotype_name == self.genotype_names[0]
                        else None,
                    ),
                    hovertemplate=f"<b>{genotype_name}</b><br>"
                    + "Branching: %{x:.3f}<br>"
                    + "Limbs: %{y:.3f}<br>"
                    + "Fitness: %{marker.color:.3f}<br>"
                    + "<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Individual Analysis - Generation {generation}<br><sub>Click on any point to download robot</sub>",
            xaxis_title="Branching",
            yaxis_title="Limbs",
            height=600,
            showlegend=True,
        )

        return dcc.Graph(id="individual-scatter", figure=fig)

    def run(self, host="127.0.0.1", port=8051, debug=True):
        """Run the dashboard server."""
        print(
            f"Starting Comparative Evolution Dashboard at http://{host}:{port}"
        )
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def load_config_from_saved_json(config_path: Path) -> Any:
    """Load configuration from saved JSON file created by evolve_headless."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
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
            self.mutation_name = resolved.get("mutation_name", "unknown")
            self.crossover_name = resolved.get("crossover_name", "unknown")
            # Store the full config data for reference
            self._full_config = config_data

    return DashboardConfig()


def load_config_for_genotype(config_path: Path) -> Any:
    """Load configuration for a specific genotype (fallback to TOML)."""
    cfg = tomllib.loads(config_path.read_text())

    gname = cfg["run"]["genotype"]
    task = cfg["run"]["task"]
    task_params = cfg["task"].get(task, {}).get("params", {})
    genotype = GENOTYPES_MAPPING[gname]

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
            self.mutation_name = "unknown"
            self.crossover_name = "unknown"
            self.output_folder = Path(cfg["data"]["output_folder"])
            self.db_file_name = cfg["data"]["db_file_name"]
            self.db_file_path = (
                Path(cfg["data"]["output_folder"]) / cfg["data"]["db_file_name"]
            )

    return DashboardConfig()


def load_populations_from_database(
    db_path: Path, genotype_name: str
) -> Tuple[List[Population], Any]:
    """Load populations from a database for a specific genotype."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    console.log(f"Loading {genotype_name} data from: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")
    populations = []

    with Session(engine) as session:
        result = session.exec(select(Individual.time_of_birth)).all()
        if not result:
            raise ValueError(f"No individuals found in database: {db_path}")

        generations = sorted(set(result))
        console.log(f"{genotype_name}: Found {len(generations)} generations")

        for gen in generations:
            individuals = session.exec(
                select(Individual).where(Individual.time_of_birth == gen)
            ).all()

            population = [Individual(**ind.model_dump()) for ind in individuals]
            populations.append(population)

    # Load config - first try the saved JSON config file
    config_filename = db_path.stem + "_config.json"
    json_config_path = db_path.parent / config_filename

    if json_config_path.exists():
        console.log(f"Loading saved configuration from: {json_config_path}")
        config = load_config_from_saved_json(json_config_path)
    else:
        # Fallback to TOML config files
        console.log(
            f"No saved config found at {json_config_path}, trying TOML fallback"
        )
        config_path = db_path.parent / "config.toml"
        if not config_path.exists():
            # Try examples/config.toml as fallback
            config_path = Path("examples/config.toml")

        if config_path.exists():
            console.log(f"Loading TOML configuration from: {config_path}")
            config = load_config_for_genotype(config_path)
        else:
            raise FileNotFoundError(
                f"No configuration file found for database {db_path}"
            )

    return populations, config


def main():
    """Main entry point for comparative dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comparative evolution dashboard"
    )
    parser.add_argument(
        "--db_paths",
        nargs="+",
        help="Paths to database files",
        default=["__data__/tree.db", "__data__/lsystem.db"],
        required=False,
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names for genotypes (default: genotype names from config)",
        default=["Tree", "L-System"],
        required=False,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8051, help="Dashboard port")
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable debug mode"
    )

    args = parser.parse_args()

    if len(args.db_paths) > 3:
        console.log(
            "Warning: Only first 3 databases will be used for comparison"
        )
        args.db_paths = args.db_paths[:3]

    genotype_data = {}

    for i, db_path in enumerate(args.db_paths):
        db_path = Path(db_path)

        try:
            populations, config = load_populations_from_database(
                db_path, f"genotype_{i}"
            )

            if args.names and i < len(args.names):
                genotype_name = args.names[i]
            else:
                # Use the genotype name from the saved config if available
                if hasattr(config, "genotype_name"):
                    genotype_name = config.genotype_name.capitalize()
                else:
                    genotype_name = config.genotype.__name__.replace(
                        "Genotype", ""
                    )

            genotype_data[genotype_name] = (populations, config)
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
            console.log(f"Error loading {db_path}: {e}")
            continue

    if not genotype_data:
        console.log("No valid databases loaded!")
        return

    console.log(
        f"Starting comparative dashboard with {len(genotype_data)} genotypes"
    )
    dashboard = ComparativeEvolutionDashboard(genotype_data)
    dashboard.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()
