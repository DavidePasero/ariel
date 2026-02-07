#!/usr/bin/env python3
"""
Novelty Search Dashboard

This module provides an interactive web dashboard for visualizing novelty
search evolutionary computation results using Plotly Dash. It displays
morphological space coverage, archive evolution, and diversity metrics.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Callable, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
from experiments.genomes.metrics import compute_6d_descriptor

type Population = List[Individual]


class NoveltySearchDashboard:
    """Interactive dashboard for novelty search visualization."""

    def __init__(
        self, populations: List[Population], decoder: Callable, config: Any
    ):
        """Initialize dashboard with novelty search evolution data.

        Args:
            populations: List of populations per generation
            decoder: Function to decode Individual genotype to robot graph
            config: Evolution configuration object
        """
        self.populations = populations[:-1]
        self.decoder = decoder
        self.config = config
        self.analyzer = PlotlyMorphologyAnalyzer()

        # Pre-compute novelty timeline
        self._compute_novelty_timeline()

        # Pre-compute all descriptors for archive
        self._compute_all_descriptors()

        # Cache for PCA/t-SNE projections
        self._projection_cache = {}

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

    def _compute_novelty_timeline(self):
        """Compute novelty statistics for each generation."""
        self.novelty_timeline = []
        self.archive_timeline = []

        cumulative_archive = []

        for gen_idx, population in enumerate(self.populations):
            if population:
                novelty_scores = [ind.fitness for ind in population]

                self.novelty_timeline.append({
                    "generation": gen_idx,
                    "mean_novelty": np.mean(novelty_scores),
                    "std_novelty": np.std(novelty_scores),
                    "max_novelty": max(novelty_scores),
                    "min_novelty": min(novelty_scores),
                    "median_novelty": np.median(novelty_scores),
                    "population_size": len(population),
                })

                # Track archive (all individuals seen so far)
                cumulative_archive.extend(population)
                self.archive_timeline.append({
                    "generation": gen_idx,
                    "archive_size": len(cumulative_archive),
                })

    def _compute_all_descriptors(self):
        """Pre-compute morphological descriptors for all individuals."""
        self.all_descriptors = []
        self.all_individuals = []
        self.descriptor_generations = []

        for gen_idx, population in enumerate(self.populations):
            for ind in population:
                robot_graph = self.decoder(ind)
                descriptor = compute_6d_descriptor(robot_graph)

                self.all_descriptors.append(descriptor)
                self.all_individuals.append(ind)
                self.descriptor_generations.append(gen_idx)

        self.all_descriptors = np.array(self.all_descriptors)
        self.descriptor_generations = np.array(self.descriptor_generations)

    def _get_projection(self, method="pca", n_components=2):
        """Get or compute dimensionality reduction projection.

        Args:
            method: 'pca' or 'tsne'
            n_components: Number of dimensions (2 or 3)

        Returns:
            Projected coordinates
        """
        cache_key = f"{method}_{n_components}"

        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        else:  # tsne
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(self.all_descriptors) - 1),
            )

        projection = reducer.fit_transform(self.all_descriptors)
        self._projection_cache[cache_key] = projection

        return projection

    def _setup_layout(self):
        """Setup the dashboard layout."""
        max_generation = len(self.populations) - 2

        self.app.layout = html.Div([
            html.H1(
                "Novelty Search Dashboard",
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
            # Novelty timeline (always visible)
            html.Div(
                [
                    html.H3("Novelty Scores Over Generations"),
                    dcc.Graph(id="novelty-timeline"),
                ],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Archive growth
            html.Div(
                [html.H3("Archive Growth"), dcc.Graph(id="archive-growth")],
                style={"margin": "20px", "marginBottom": 40},
            ),
            # Fixed Robot Viewer
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
                    html.Div(
                        id="robot-viewer",
                        children=[
                            html.P(
                                "Click on any plot to view robots here",
                                style={
                                    "textAlign": "center",
                                    "color": "#666",
                                    "fontSize": 14,
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
                        value="space-coverage-tab",
                        children=[
                            dcc.Tab(
                                label="Morphological Space Coverage",
                                value="space-coverage-tab",
                            ),
                            dcc.Tab(
                                label="Feature Distributions",
                                value="distributions-tab",
                            ),
                            dcc.Tab(
                                label="Feature Evolution", value="evolution-tab"
                            ),
                            dcc.Tab(
                                label="Diversity Analysis",
                                value="diversity-tab",
                            ),
                            dcc.Tab(
                                label="K-Nearest Neighbors", value="knn-tab"
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
            Output("novelty-timeline", "figure"),
            Input("generation-slider", "value"),
        )
        def update_novelty_timeline(selected_generation):
            """Update novelty timeline with highlighted generation."""
            if not self.novelty_timeline:
                return go.Figure()

            df = pd.DataFrame(self.novelty_timeline)

            fig = go.Figure()

            # Add mean novelty line
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["mean_novelty"],
                    mode="lines+markers",
                    name="Mean Novelty",
                    line=dict(color="blue", width=2),
                )
            )

            # Add std deviation band
            fig.add_trace(
                go.Scatter(
                    x=df["generation"].tolist()
                    + df["generation"][::-1].tolist(),
                    y=(df["mean_novelty"] + df["std_novelty"]).tolist()
                    + (df["mean_novelty"] - df["std_novelty"])[::-1].tolist(),
                    fill="toself",
                    fillcolor="rgba(0,100,200,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=True,
                    name="±1 Std Dev",
                )
            )

            # Add max novelty line
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["max_novelty"],
                    mode="lines+markers",
                    name="Max Novelty",
                    line=dict(color="green", width=2),
                )
            )

            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(
                    go.Scatter(
                        x=[selected_row["generation"]],
                        y=[selected_row["mean_novelty"]],
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
                title="Novelty Score Evolution<br><sub>Click on any point to view the most novel robot</sub>",
                xaxis_title="Generation",
                yaxis_title="Novelty Score",
                height=400,
                showlegend=True,
            )

            return fig

        @self.app.callback(
            Output("archive-growth", "figure"),
            Input("generation-slider", "value"),
        )
        def update_archive_growth(selected_generation):
            """Update archive growth plot."""
            if not self.archive_timeline:
                return go.Figure()

            df = pd.DataFrame(self.archive_timeline)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["archive_size"],
                    mode="lines+markers",
                    name="Archive Size",
                    line=dict(color="purple", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(128,0,128,0.2)",
                )
            )

            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(
                    go.Scatter(
                        x=[selected_row["generation"]],
                        y=[selected_row["archive_size"]],
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
                title="Archive Growth Over Time",
                xaxis_title="Generation",
                yaxis_title="Total Individuals in Archive",
                height=400,
                showlegend=True,
            )

            return fig

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("plot-tabs", "value"), Input("generation-slider", "value")],
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            if active_tab == "space-coverage-tab":
                return self._create_space_coverage_plot(selected_generation)
            elif active_tab == "distributions-tab":
                return self._create_distributions_plot(selected_generation)
            elif active_tab == "evolution-tab":
                return self._create_feature_evolution_plot()
            elif active_tab == "diversity-tab":
                return self._create_diversity_plot(selected_generation)
            elif active_tab == "knn-tab":
                return self._create_knn_plot(selected_generation)

            return html.Div("Select a tab to view plots")

        @self.app.callback(
            [
                Output("status-message", "children"),
                Output("status-message", "style"),
            ],
            Input("novelty-timeline", "clickData"),
            prevent_initial_call=True,
        )
        def handle_novelty_click(clickData):
            """Handle clicks on novelty timeline."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                point = clickData["points"][0]
                generation = int(point["x"])
                return self._handle_generation_click(generation, mode="max")
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
            Output("robot-viewer", "children"),
            Input("novelty-timeline", "clickData"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_novelty(clickData):
            """Update robot viewer when novelty timeline is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        @self.app.callback(
            Output("robot-viewer", "children", allow_duplicate=True),
            Input("space-coverage-scatter", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def update_robot_viewer_scatter(clickData, generation):
            """Update robot viewer when space coverage scatter is clicked."""
            if clickData and self.current_robot_image is not None:
                return self._render_robot_display(self.current_robot_image)
            return dash.no_update

        @self.app.callback(
            [
                Output("status-message", "children", allow_duplicate=True),
                Output("status-message", "style", allow_duplicate=True),
            ],
            Input("space-coverage-scatter", "clickData"),
            State("generation-slider", "value"),
            prevent_initial_call=True,
        )
        def handle_scatter_click(clickData, generation):
            """Handle clicks on space coverage scatter."""
            if not clickData or "points" not in clickData:
                return "", {"display": "none"}

            try:
                point = clickData["points"][0]
                point_index = point.get(
                    "pointIndex", point.get("pointNumber", 0)
                )

                # Get the individual
                individual = self.all_individuals[point_index]
                robot_graph = self.decoder(individual)

                # Save robot
                robot_data = json_graph.node_link_data(
                    robot_graph, edges="edges"
                )
                gen = self.descriptor_generations[point_index]
                filename = (
                    f"robot_gen{gen}_novelty{individual.fitness:.3f}.json"
                )
                filepath = self.dl_robots_path / filename

                with open(filepath, "w") as f:
                    json.dump(robot_data, f, indent=2)

                # Render the robot
                rendered_img = self._render_robot(robot_graph)

                if rendered_img:
                    self.current_robot_image = {
                        "image": rendered_img,
                        "generation": gen,
                        "novelty": individual.fitness,
                        "filename": filename,
                    }

                return f"Robot saved: {filename}", {
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

    def _create_space_coverage_plot(self, selected_generation):
        """Create morphological space coverage visualization."""
        # Controls for projection method
        controls = html.Div([
            html.Div(
                [
                    html.Label(
                        "Projection Method:",
                        style={"fontWeight": "bold", "marginRight": 10},
                    ),
                    dcc.RadioItems(
                        id="projection-method",
                        options=[
                            {"label": "PCA", "value": "pca"},
                            {"label": "t-SNE", "value": "tsne"},
                        ],
                        value="pca",
                        inline=True,
                        style={"display": "inline-block"},
                    ),
                ],
                style={"marginBottom": 20},
            ),
        ])

        # Get PCA projection (default)
        projection = self._get_projection("pca", 2)

        # Create scatter plot
        fig = go.Figure()

        # Color by generation
        fig.add_trace(
            go.Scatter(
                x=projection[:, 0],
                y=projection[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=self.descriptor_generations,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Generation"),
                    opacity=0.7,
                ),
                text=[
                    f"Gen: {g}<br>Novelty: {ind.fitness:.3f}"
                    for g, ind in zip(
                        self.descriptor_generations, self.all_individuals
                    )
                ],
                hoverinfo="text",
            )
        )

        # Highlight current generation
        current_gen_mask = self.descriptor_generations == selected_generation
        if current_gen_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=projection[current_gen_mask, 0],
                    y=projection[current_gen_mask, 1],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="circle-open",
                        line=dict(width=2),
                    ),
                    name=f"Generation {selected_generation}",
                    showlegend=True,
                )
            )

        fig.update_layout(
            title=f"Morphological Space Coverage (PCA)<br><sub>Click on any point to view that robot</sub>",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=600,
            hovermode="closest",
        )

        return html.Div([
            controls,
            dcc.Graph(id="space-coverage-scatter", figure=fig),
        ])

    def _create_distributions_plot(self, selected_generation):
        """Create feature distribution plots."""
        # Get descriptors for selected generation
        gen_mask = self.descriptor_generations == selected_generation
        gen_descriptors = self.all_descriptors[gen_mask]

        # Create subplots for each feature
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=3, subplot_titles=self.feature_names)

        for i, feature_name in enumerate(self.feature_names):
            row = i // 3 + 1
            col = i % 3 + 1

            # Current generation histogram
            fig.add_trace(
                go.Histogram(
                    x=gen_descriptors[:, i],
                    name=f"Gen {selected_generation}",
                    opacity=0.7,
                    nbinsx=20,
                    marker_color="blue",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

            # All archive histogram
            fig.add_trace(
                go.Histogram(
                    x=self.all_descriptors[:, i],
                    name="All Archive",
                    opacity=0.5,
                    nbinsx=20,
                    marker_color="gray",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title=f"Feature Distributions - Generation {selected_generation}",
            height=800,
            showlegend=True,
            barmode="overlay",
        )

        return dcc.Graph(figure=fig)

    def _create_feature_evolution_plot(self):
        """Create feature evolution over time."""
        # Compute statistics per generation
        evolution_data = []

        for gen_idx in range(len(self.populations)):
            gen_mask = self.descriptor_generations == gen_idx
            gen_descriptors = self.all_descriptors[gen_mask]

            if len(gen_descriptors) > 0:
                for feat_idx, feat_name in enumerate(self.feature_names):
                    evolution_data.append({
                        "generation": gen_idx,
                        "feature": feat_name,
                        "mean": np.mean(gen_descriptors[:, feat_idx]),
                        "std": np.std(gen_descriptors[:, feat_idx]),
                        "min": np.min(gen_descriptors[:, feat_idx]),
                        "max": np.max(gen_descriptors[:, feat_idx]),
                    })

        df = pd.DataFrame(evolution_data)

        fig = go.Figure()

        for feat_name in self.feature_names:
            feat_data = df[df["feature"] == feat_name]

            fig.add_trace(
                go.Scatter(
                    x=feat_data["generation"],
                    y=feat_data["mean"],
                    mode="lines",
                    name=feat_name,
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title="Feature Evolution Over Generations",
            xaxis_title="Generation",
            yaxis_title="Feature Value (normalized)",
            height=500,
            hovermode="x unified",
        )

        return dcc.Graph(figure=fig)

    def _create_diversity_plot(self, selected_generation):
        """Create diversity analysis plot."""
        # Compute diversity metrics per generation
        diversity_data = []

        for gen_idx in range(len(self.populations)):
            gen_mask = self.descriptor_generations == gen_idx
            gen_descriptors = self.all_descriptors[gen_mask]

            if len(gen_descriptors) > 0:
                # Compute diversity as std dev across features
                diversities = [
                    np.std(gen_descriptors[:, i])
                    for i in range(len(self.feature_names))
                ]

                diversity_data.append({
                    "generation": gen_idx,
                    "diversities": diversities,
                })

        # Create radar chart for selected generation
        selected_data = next(
            (
                d
                for d in diversity_data
                if d["generation"] == selected_generation
            ),
            None,
        )

        if selected_data:
            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=selected_data["diversities"],
                    theta=self.feature_names,
                    fill="toself",
                    name=f"Generation {selected_generation}",
                )
            )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(selected_data["diversities"]) * 1.2],
                    )
                ),
                title=f"Morphological Diversity - Generation {selected_generation}",
                height=500,
            )

            return dcc.Graph(figure=fig)

        return html.Div("No data available")

    def _create_knn_plot(self, selected_generation):
        """Create k-nearest neighbors visualization."""
        info_text = html.P(
            [
                "K-Nearest Neighbors Analysis:",
                html.Br(),
                "Select an individual from the Space Coverage plot to view its k-nearest neighbors.",
                html.Br(),
                "This shows how novelty is calculated based on distance to neighbors.",
            ],
            style={"textAlign": "center", "color": "#666", "fontSize": 14},
        )

        return html.Div([info_text])

    def _handle_generation_click(self, generation: int, mode: str = "max"):
        """Handle click on a generation plot."""
        try:
            if generation < 0 or generation >= len(self.populations):
                return "Error: Invalid generation", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                }

            population = self.populations[generation]
            if not population:
                return "Error: Empty population", {
                    "display": "block",
                    "backgroundColor": "#ffcccc",
                }

            # Find most novel individual
            if mode == "max":
                individual = max(population, key=lambda x: x.fitness)
            else:
                individual = min(population, key=lambda x: x.fitness)

            robot_graph = self.decoder(individual)

            # Save robot
            robot_data = json_graph.node_link_data(robot_graph, edges="edges")
            filename = (
                f"robot_gen{generation}_novelty{individual.fitness:.3f}.json"
            )
            filepath = self.dl_robots_path / filename

            with open(filepath, "w") as f:
                json.dump(robot_data, f, indent=2)

            # Render the robot
            rendered_img = self._render_robot(robot_graph)

            if rendered_img:
                self.current_robot_image = {
                    "image": rendered_img,
                    "generation": generation,
                    "novelty": individual.fitness,
                    "filename": filename,
                }

            return f"Robot saved: {filename}", {
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
        """Render a robot using MuJoCo and return base64-encoded image."""
        try:
            mj.set_mjcb_control(None)
            robot_spec = construct_mjspec_from_graph(robot_graph)
            world = OlympicArena()
            spawn_pos = [0, 0, 0.1]
            world.spawn(robot_spec.spec, spawn_position=spawn_pos)

            model = world.spec.compile()
            data = mj.MjData(model)

            camera = mj.MjvCamera()
            camera.type = mj.mjtCamera.mjCAMERA_FREE
            camera.lookat = [0, 0, 0.3]
            camera.distance = 2.5
            camera.azimuth = 45
            camera.elevation = -20

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

            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return img_str

        except Exception as e:
            print(f"Error rendering robot: {str(e)}")
            return None

    def _render_robot_display(self, robot_info):
        """Render robot display HTML."""
        return html.Div([
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
                            html.Strong("Novelty Score: "),
                            html.Span(f"{robot_info['novelty']:.4f}"),
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

    def run(self, host="127.0.0.1", port=8052, debug=True):
        """Run the dashboard server."""
        print(f"Starting Novelty Search Dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def run_dashboard(
    populations: List[Population],
    decoder: Callable,
    config: Any,
    host="127.0.0.1",
    port=8052,
    debug=True,
):
    """Run the novelty search dashboard."""
    dashboard = NoveltySearchDashboard(populations, decoder, config)
    dashboard.run(host=host, port=port, debug=debug)
