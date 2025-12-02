#!/usr/bin/env python3
"""
Interactive Evolution Dashboard

This module provides an interactive web dashboard for visualizing evolutionary
computation results using Plotly Dash. It displays morphological analysis
plots from MorphologyAnalyzer with generation selection capabilities.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Callable, Any
from networkx.readwrite import json_graph
import mujoco as mj

from experiments.genomes.plotly_morphology_analysis import (
    PlotlyMorphologyAnalyzer,
)
from ariel.ec.a001 import Individual
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer

type Population = List[Individual]


class EvolutionDashboard:
    """Interactive dashboard for evolution visualization."""

    def __init__(
        self, populations: List[Population], decoder: Callable, config: Any
    ):
        """Initialize dashboard with evolution data.

        Args:
            populations: List of populations per generation
            decoder: Function to decode Individual genotype to robot graph
            config: Evolution configuration object
        """
        self.populations = populations
        self.decoder = decoder
        self.config = config
        self.analyzer = PlotlyMorphologyAnalyzer()

        # Load target robots
        self.target_robot_path = Path(config.task_params["target_robot_path"])
        self.analyzer.load_target_robots(str(self.target_robot_path))

        # Load target robot graphs for rendering
        self._load_target_robot_graphs()

        # Pre-compute fitness timeline
        self._compute_fitness_timeline()

        # Cache for computed descriptors per generation
        self._descriptor_cache = {}

        # Cache for individual data per generation (for click handling)
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

        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()

    def _load_target_robot_graphs(self):
        """Load target robot graphs from JSON files."""
        self.target_robot_graphs = {}

        if self.target_robot_path.is_file():
            # Single target robot file
            target_files = [self.target_robot_path]
        elif self.target_robot_path.is_dir():
            # Directory of target robot files
            target_files = sorted(self.target_robot_path.glob("*.json"))
        else:
            target_files = []

        for target_file in target_files:
            try:
                with open(target_file, "r") as f:
                    robot_data = json.load(f)
                    robot_graph = json_graph.node_link_graph(
                        robot_data, edges="edges"
                    )
                    # Use filename without extension as key
                    robot_name = target_file.stem
                    self.target_robot_graphs[robot_name] = robot_graph
            except Exception as e:
                print(
                    f"Warning: Could not load target robot {target_file}: {e}"
                )

    def _compute_fitness_timeline(self):
        """Compute average fitness for each generation."""
        self.fitness_timeline = []

        for gen_idx, population in enumerate(self.populations):
            if population:
                avg_fitness = sum(ind.fitness for ind in population) / len(
                    population
                )
                self.fitness_timeline.append({
                    "generation": gen_idx,
                    "avg_fitness": avg_fitness,
                    "best_fitness": max(ind.fitness for ind in population),
                    "worst_fitness": min(ind.fitness for ind in population),
                })

    def _get_generation_data(self, generation: int):
        """Get or compute morphological data for a specific generation."""
        if generation in self._descriptor_cache:
            return self._descriptor_cache[generation]

        if generation >= len(self.populations):
            generation = len(self.populations) - 1

        # Load population data into analyzer
        population = self.populations[generation]
        self.analyzer.load_population(population, self.decoder)
        self.analyzer.compute_fitness_scores()

        # Cache the results
        self._descriptor_cache[generation] = {
            "descriptors": self.analyzer.descriptors.copy(),
            "fitness_scores": self.analyzer.fitness_scores.copy(),
        }

        # Cache individual data for click handling
        self._individual_cache[generation] = population.copy()

        return self._descriptor_cache[generation]

    def _setup_layout(self):
        """Setup the dashboard layout."""
        max_generation = len(self.populations) - 1

        self.app.layout = html.Div([
            html.H1(
                "Evolution Dashboard",
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
            # Fitness timeline (always visible)
            html.Div(
                [
                    html.H3("Fitness Over Generations"),
                    dcc.Graph(id="fitness-timeline"),
                ],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Fixed Robot Viewer (always visible, updates on click)
            html.Div(
                [
                    html.H3(
                        "Robot Viewer",
                        style={
                            "textAlign": "center",
                            "marginTop": 20,
                            "marginBottom": 20,
                        },
                    ),
                    # Target robot selector
                    html.Div(
                        [
                            html.Label(
                                "View Target Robot:",
                                style={"fontWeight": "bold", "marginRight": 10},
                            ),
                            dcc.Dropdown(
                                id="target-robot-dropdown",
                                options=[
                                    {"label": name, "value": name}
                                    for name in self.target_robot_graphs.keys()
                                ],
                                placeholder="Select a target robot to view...",
                                style={
                                    "width": "300px",
                                    "display": "inline-block",
                                },
                            ),
                        ],
                        style={"textAlign": "center", "marginBottom": 20},
                    )
                    if self.target_robot_graphs
                    else None,
                    html.Div(
                        id="fixed-robot-viewer",
                        children=[
                            html.P(
                                [
                                    "Click on robots to view them here:",
                                    html.Br(),
                                    "• Fitness Timeline → Best robot from generation",
                                    html.Br(),
                                    "• Single Feature plots → Robot closest to mean fitness",
                                    html.Br(),
                                    "• Fitness Landscapes/Pairwise Features → Specific robot",
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
                                label="Single Feature",
                                value="single-feature-tab",
                            ),
                            dcc.Tab(
                                label="Fitness Landscapes",
                                value="landscape-tab",
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
                                label="Pairwise Features", value="pairwise-tab"
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
            Output("fitness-timeline", "figure"),
            Input("generation-slider", "value"),
        )
        def update_fitness_timeline(selected_generation):
            """Update fitness timeline with highlighted generation."""
            if not self.fitness_timeline:
                return go.Figure()

            df = pd.DataFrame(self.fitness_timeline)

            fig = go.Figure()

            # Add average fitness line
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["avg_fitness"],
                    mode="lines+markers",
                    name="Average Fitness",
                    line=dict(color="blue", width=2),
                )
            )

            # Add best fitness line
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["best_fitness"],
                    mode="lines+markers",
                    name="Best Fitness",
                    line=dict(color="green", width=2),
                )
            )

            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(
                    go.Scatter(
                        x=[selected_row["generation"]],
                        y=[selected_row["avg_fitness"]],
                        mode="markers",
                        name=f"Generation {selected_generation}",
                        marker=dict(
                            color="red",
                            size=12,
                            symbol="circle-open",
                            line=dict(width=3),
                        ),
                    )
                )

            fig.update_layout(
                title="Fitness Evolution Over Generations<br><sub>Click on any point to view the best robot from that generation</sub>",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=400,
                showlegend=True,
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
            return self._plot_feature_evolution(feature_idx)

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("plot-tabs", "value"), Input("generation-slider", "value")],
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            # Get data for selected generation
            gen_data = self._get_generation_data(selected_generation)

            if active_tab == "landscape-tab":
                return self._create_landscape_plot(selected_generation)
            elif active_tab == "distribution-tab":
                return self._create_distribution_plot(selected_generation)
            elif active_tab == "diversity-tab":
                return self._create_diversity_plot(selected_generation)
            elif active_tab == "pairwise-tab":
                return self._create_pairwise_plot(selected_generation)
            elif active_tab == "single-feature-tab":
                return self._create_single_feature_plot()
            elif active_tab == "robot-viewer-tab":
                return self._create_robot_viewer()

            return html.Div("Select a tab to view plots")

        # Separate callbacks for each clickable graph
        @self.app.callback(
            Output("status-message", "children", allow_duplicate=True),
            Output("status-message", "style", allow_duplicate=True),
            Input("landscape-graph", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def handle_landscape_click(clickData, generation):
            """Handle clicks on landscape graph."""
            if not clickData:
                return "", {"display": "none"}
            return self._handle_point_click(clickData, generation)

        @self.app.callback(
            Output("status-message", "children", allow_duplicate=True),
            Output("status-message", "style", allow_duplicate=True),
            Input("pairwise-graph", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def handle_pairwise_click(clickData, generation):
            """Handle clicks on pairwise graph."""
            if not clickData:
                return "", {"display": "none"}
            return self._handle_point_click(clickData, generation)

        @self.app.callback(
            Output("status-message", "children", allow_duplicate=True),
            Output("status-message", "style", allow_duplicate=True),
            Input("fitness-timeline", "clickData"),
            prevent_initial_call=True,
        )
        def handle_fitness_timeline_click(clickData):
            """Handle clicks on fitness timeline - renders best robot from clicked generation."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                # Get clicked generation
                point = clickData["points"][0]
                generation = int(point["x"])

                # Get best robot from that generation
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

        @self.app.callback(
            Output("status-message", "children", allow_duplicate=True),
            Output("status-message", "style", allow_duplicate=True),
            Input("single-feature-graph", "clickData"),
            prevent_initial_call=True,
        )
        def handle_single_feature_click(clickData):
            """Handle clicks on single feature graph - renders robot closest to mean fitness."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                # Get clicked generation
                point = clickData["points"][0]
                generation = int(point["x"])

                # Get robot closest to mean fitness from that generation
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

        # Callback for target robot dropdown
        @self.app.callback(
            Output("fixed-robot-viewer", "children", allow_duplicate=True),
            Input("target-robot-dropdown", "value"),
            prevent_initial_call=True,
        )
        def display_target_robot(target_name):
            """Display selected target robot."""
            if not target_name or target_name not in self.target_robot_graphs:
                return dash.no_update

            # Get the target robot graph
            robot_graph = self.target_robot_graphs[target_name]

            # Render the target robot
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

        # Callback to update fixed robot viewer for timeline clicks
        @self.app.callback(
            Output("fixed-robot-viewer", "children", allow_duplicate=True),
            Input("fitness-timeline", "clickData"),
            prevent_initial_call=True,
        )
        def update_fixed_robot_viewer_timeline(timeline_click):
            """Update the fixed robot viewer when fitness timeline is clicked."""
            if timeline_click and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Separate callback for dynamically created single feature graph
        @self.app.callback(
            Output("fixed-robot-viewer", "children", allow_duplicate=True),
            Input("single-feature-graph", "clickData"),
            prevent_initial_call=True,
        )
        def update_fixed_robot_viewer_feature(clickData):
            """Update the fixed robot viewer when single feature graph is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Separate callback for dynamically created landscape graph
        @self.app.callback(
            Output("fixed-robot-viewer", "children", allow_duplicate=True),
            Input("landscape-graph", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def update_fixed_robot_viewer_landscape(clickData, generation):
            """Update the fixed robot viewer when landscape graph is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        # Separate callback for dynamically created pairwise graph
        @self.app.callback(
            Output("fixed-robot-viewer", "children", allow_duplicate=True),
            Input("pairwise-graph", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def update_fixed_robot_viewer_pairwise(clickData, generation):
            """Update the fixed robot viewer when pairwise graph is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

    def _create_landscape_plot(self, generation: int):
        """Create fitness landscape plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_fitness_landscapes()
        fig.update_layout(
            title_text=f"Fitness Landscapes - Generation {generation}"
        )

        # Create graph with click handling
        return dcc.Graph(id="landscape-graph", figure=fig)

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

    def _handle_generation_click(self, generation: int, mode: str = "best"):
        """Handle click on a generation plot (fitness timeline or single feature).

        Args:
            generation: Generation index
            mode: 'best' for best individual, 'mean' for closest to mean fitness

        Returns:
            Status message and style dict
        """
        try:
            # Validate generation
            if generation < 0 or generation >= len(self.populations):
                return "Error: Invalid generation", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

            # Get population
            population = self.populations[generation]
            if not population:
                return "Error: Empty population", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

            # Find the target individual based on mode
            if mode == "best":
                # Get best individual (highest fitness)
                point_index = max(
                    range(len(population)), key=lambda i: population[i].fitness
                )
                label = "best"
            else:  # mode == 'mean'
                # Get individual closest to mean fitness
                fitnesses = [ind.fitness for ind in population]
                mean_fitness = np.mean(fitnesses)
                point_index = min(
                    range(len(population)),
                    key=lambda i: abs(population[i].fitness - mean_fitness),
                )
                label = f"closest to mean (mean={mean_fitness:.3f})"

            # Get the individual and convert to robot graph
            individual = population[point_index]
            robot_graph = self.decoder(individual)

            # Save robot as JSON
            robot_data = json_graph.node_link_data(robot_graph, edges="edges")

            # Save to file
            filename = f"robot_gen{generation}_{mode}_ind{point_index}_fit{individual.fitness:.3f}.json"
            filepath = self.dl_robots_path / filename

            with open(filepath, "w") as f:
                json.dump(robot_data, f, indent=2)

            # Render the robot
            rendered_img = self._render_robot(robot_graph)

            # Store the rendered robot for the viewer tab
            if rendered_img:
                self.current_robot_image = {
                    "image": rendered_img,
                    "generation": generation,
                    "individual": point_index,
                    "fitness": individual.fitness,
                    "filename": filename,
                    "mode": mode,
                }

            message = f"Robot saved: {filename} ({label})"
            if rendered_img:
                message += " | View in 'Robot Viewer' tab →"

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

    def _handle_point_click(self, clickData, generation):
        """Handle click on a data point to save robot and render it."""
        if not clickData or "points" not in clickData:
            return "", {"display": "none"}

        try:
            # Get the clicked point index
            point = clickData["points"][0]
            point_index = point.get("pointIndex", point.get("pointNumber", 0))

            # Get the individual from cache
            if generation not in self._individual_cache:
                return "Error: Generation data not found", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

            population = self._individual_cache[generation]
            if point_index >= len(population):
                return f"Error: Point index {point_index} out of range", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                    "color": "red",
                    "textAlign": "center",
                    "marginBottom": 20,
                    "padding": "10px",
                    "borderRadius": "5px",
                }

            # Get the individual and convert to robot graph
            individual = population[point_index]
            robot_graph = self.decoder(individual)

            # Save robot as JSON
            robot_data = json_graph.node_link_data(robot_graph, edges="edges")

            # Save to file
            filename = f"robot_gen{generation}_ind{point_index}_fit{individual.fitness:.3f}.json"
            filepath = self.dl_robots_path / filename

            with open(filepath, "w") as f:
                json.dump(robot_data, f, indent=2)

            # Render the robot
            rendered_img = self._render_robot(robot_graph)

            # Store the rendered robot for the viewer tab
            if rendered_img:
                self.current_robot_image = {
                    "image": rendered_img,
                    "generation": generation,
                    "individual": point_index,
                    "fitness": individual.fitness,
                    "filename": filename,
                }

            message = f"Robot saved: {filename}"
            if rendered_img:
                message += " | View in 'Robot Viewer' tab →"

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

    def _create_distribution_plot(self, generation: int):
        """Create fitness distribution plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_fitness_distributions()
        fig.update_layout(
            title_text=f"Fitness Distributions - Generation {generation}"
        )
        return dcc.Graph(figure=fig)

    def _create_diversity_plot(self, generation: int):
        """Create morphological diversity analysis plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.analyze_morphological_diversity()
        fig.update_layout(
            title_text=f"Morphological Diversity - Generation {generation}"
        )
        return dcc.Graph(figure=fig)

    def _create_pairwise_plot(self, generation: int):
        """Create pairwise feature landscape plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_pairwise_feature_landscapes()
        fig.update_layout(
            title_text=f"Pairwise Feature Landscapes - Generation {generation}"
        )

        # Create graph with click handling
        return dcc.Graph(id="pairwise-graph", figure=fig)

    def _create_single_feature_plot(self):
        """Create single feature evolution plot with feature selector."""
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

    def _plot_feature_evolution(self, feature_idx: int):
        """Plot evolution of a single morphological feature over generations."""
        if feature_idx is None:
            return go.Figure()

        feature_name = self.feature_names[feature_idx]

        # Compute feature values for all generations
        generation_data = []

        for gen_idx in range(len(self.populations)):
            gen_data = self._get_generation_data(gen_idx)
            if gen_data["descriptors"].size > 0:
                # Extract the specific feature for this generation
                feature_values = gen_data["descriptors"][:, feature_idx]

                # Find best individual (closest to any target)
                best_value = feature_values[0]
                if (
                    hasattr(self.analyzer, "target_descriptors")
                    and len(self.analyzer.target_descriptors) > 0
                ):
                    # Find individual closest to any target for this feature
                    min_distance = float("inf")
                    for target_desc in self.analyzer.target_descriptors:
                        target_value = target_desc[feature_idx]
                        distances = np.abs(feature_values - target_value)
                        closest_idx = np.argmin(distances)
                        if distances[closest_idx] < min_distance:
                            min_distance = distances[closest_idx]
                            best_value = feature_values[closest_idx]

                generation_data.append({
                    "generation": gen_idx,
                    "mean": np.mean(feature_values),
                    "std": np.std(feature_values),
                    "min": np.min(feature_values),
                    "max": np.max(feature_values),
                    "median": np.median(feature_values),
                    "best": best_value,
                })

        if not generation_data:
            return go.Figure()

        df = pd.DataFrame(generation_data)

        fig = go.Figure()

        # Add mean line with error bands
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["mean"],
                mode="lines+markers",
                name="Population Mean",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        # Add standard deviation bands
        fig.add_trace(
            go.Scatter(
                x=df["generation"].tolist() + df["generation"][::-1].tolist(),
                y=(df["mean"] + df["std"]).tolist()
                + (df["mean"] - df["std"])[::-1].tolist(),
                fill="tonext",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="±1 Standard Deviation",
            )
        )

        # Add median line
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["median"],
                mode="lines+markers",
                name="Population Median",
                line=dict(color="purple", width=2, dash="dot"),
                marker=dict(size=4),
            )
        )

        # Add population best line (closest to target)
        fig.add_trace(
            go.Scatter(
                x=df["generation"],
                y=df["best"],
                mode="lines+markers",
                name="Population Best (Closest to Target)",
                line=dict(color="green", width=2),
                marker=dict(size=5, symbol="star"),
            )
        )

        # Add target robot values as horizontal lines
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
            title=f"{feature_name} Evolution Over Generations<br><sub>Click on any point to view the robot closest to mean fitness from that generation</sub>",
            xaxis_title="Generation",
            yaxis_title=f"{feature_name} Value",
            height=500,
            showlegend=True,
            hovermode="x unified",
        )

        return fig

    def _create_robot_viewer(self):
        """Create robot viewer display."""
        if self.current_robot_image is None:
            return html.Div([
                html.H3(
                    "Robot Viewer",
                    style={"textAlign": "center", "marginTop": 40},
                ),
                html.P(
                    [
                        "Click on robots to view them here:",
                        html.Br(),
                        "• Fitness Timeline → Best robot from generation",
                        html.Br(),
                        "• Single Feature plots → Robot closest to mean fitness",
                        html.Br(),
                        "• Fitness Landscapes/Pairwise Features → Specific robot",
                    ],
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "fontSize": 14,
                        "lineHeight": "1.8",
                    },
                ),
            ])

        # Display the rendered robot
        robot_info = self.current_robot_image
        return html.Div([
            html.H3(
                f"Robot Viewer - {robot_info['filename']}",
                style={"textAlign": "center", "marginTop": 20},
            ),
            # Robot metadata
            html.Div(
                [
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
                    f"Robot saved to: {robot_info['filename']}",
                    style={
                        "textAlign": "center",
                        "color": "#666",
                        "fontSize": 12,
                        "marginTop": 20,
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

    def _render_robot_display(self, robot_info):
        """Render robot display HTML for the fixed robot viewer.

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
                )
            ]),
        ])

    def _render_target_robot_display(self, robot_info):
        """Render target robot display HTML for the fixed robot viewer.

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

    def run(self, host="127.0.0.1", port=8050, debug=True):
        """Run the dashboard server."""
        print(f"Starting Evolution Dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def run_dashboard(
    populations: List[Population],
    decoder: Callable,
    config: Any,
    host="127.0.0.1",
    port=8050,
    debug=True,
):
    """Run the evolution dashboard.

    Args:
        populations: List of populations per generation from evolution
        decoder: Function to decode Individual genotype to robot graph
        config: Evolution configuration object
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    dashboard = EvolutionDashboard(populations, decoder, config)
    dashboard.run(host=host, port=port, debug=debug)
