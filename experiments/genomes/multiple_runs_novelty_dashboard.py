#!/usr/bin/env python3
"""
Multiple Runs Novelty Search Dashboard

This module provides an interactive web dashboard for comparing novelty search
results across multiple independent runs using Plotly Dash. It displays
aggregated statistics (mean, std) for novelty scores, diversity, and space coverage.
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
from typing import List, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rich.console import Console
from sqlmodel import Session, create_engine, select
from networkx.readwrite import json_graph
import mujoco as mj

from plotly_morphology_analysis import PlotlyMorphologyAnalyzer
from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer
from experiments.genomes.metrics import compute_6d_descriptor

console = Console()
type Population = List[Individual]


class MultipleRunsNoveltyDashboard:
    """Interactive dashboard for comparing multiple novelty search runs."""

    def __init__(self, runs_data: List[Tuple[List[Population], Any]],
                 run_names: List[str] = None):
        """Initialize dashboard with multiple run data.

        Args:
            runs_data: List of (populations, config) tuples, one per run
            run_names: Optional names for each run
        """
        self.runs_data = runs_data
        self.num_runs = len(runs_data)

        if run_names is None:
            self.run_names = [f"Run {i+1}" for i in range(self.num_runs)]
        else:
            self.run_names = run_names

        # Use first run's config as reference
        _, self.config = runs_data[0]

        # Pre-compute aggregated novelty statistics
        self._compute_aggregated_novelty()

        # Pre-compute all descriptors for all runs
        self._compute_all_descriptors()

        # Morphological feature names
        self.feature_names = ['Branching', 'Limbs', 'Extensiveness',
                            'Symmetry', 'Proportion', 'Joints']

        # Create dl_robots directory
        self.dl_robots_path = Path("examples/dl_robots")
        self.dl_robots_path.mkdir(exist_ok=True)

        # Store for robot visualization
        self.current_robot_image = None

        # Color scheme for runs
        self.run_colors = px.colors.qualitative.Set2[:self.num_runs]

        # Cache for projections
        self._projection_cache = {}

        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()

    def _compute_aggregated_novelty(self):
        """Compute aggregated novelty statistics across runs."""
        self.aggregated_novelty = {}
        self.archive_growth = {}

        # Determine max generation across all runs
        max_gens = [len(populations) for populations, _ in self.runs_data]
        self.max_generation = max(max_gens)

        for gen_idx in range(self.max_generation):
            novelty_scores_across_runs = []
            archive_sizes = []

            for run_idx, (populations, config) in enumerate(self.runs_data):
                if gen_idx < len(populations):
                    population = populations[gen_idx]
                    if population:
                        novelty_scores = [ind.fitness for ind in population]
                        novelty_scores_across_runs.append({
                            'mean': np.mean(novelty_scores),
                            'max': max(novelty_scores),
                            'std': np.std(novelty_scores)
                        })

                        # Archive size is cumulative
                        archive_size = sum(len(p) for p in populations[:gen_idx+1])
                        archive_sizes.append(archive_size)

            if novelty_scores_across_runs:
                self.aggregated_novelty[gen_idx] = {
                    'mean_of_means': np.mean([r['mean'] for r in novelty_scores_across_runs]),
                    'std_of_means': np.std([r['mean'] for r in novelty_scores_across_runs]),
                    'mean_of_max': np.mean([r['max'] for r in novelty_scores_across_runs]),
                    'std_of_max': np.std([r['max'] for r in novelty_scores_across_runs]),
                }

            if archive_sizes:
                self.archive_growth[gen_idx] = {
                    'mean': np.mean(archive_sizes),
                    'std': np.std(archive_sizes),
                    'min': min(archive_sizes),
                    'max': max(archive_sizes)
                }

    def _compute_all_descriptors(self):
        """Pre-compute descriptors for all individuals in all runs."""
        self.run_descriptors = []
        self.run_individuals = []
        self.run_generations = []

        analyzer = PlotlyMorphologyAnalyzer()

        for run_idx, (populations, config) in enumerate(self.runs_data):
            descriptors = []
            individuals = []
            generations = []

            def decoder(ind):
                return config.genotype.from_json(ind.genotype).to_digraph()

            for gen_idx, population in enumerate(populations):
                for ind in population:
                    robot_graph = decoder(ind)
                    descriptor = compute_6d_descriptor(robot_graph)

                    descriptors.append(descriptor)
                    individuals.append(ind)
                    generations.append(gen_idx)

            self.run_descriptors.append(np.array(descriptors))
            self.run_individuals.append(individuals)
            self.run_generations.append(np.array(generations))

    def _color_to_rgba(self, color: str, alpha: float = 0.3) -> str:
        """Convert color to rgba with opacity."""
        if color.startswith('#'):
            rgb = px.colors.hex_to_rgb(color)
            return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'
        elif color.startswith('rgb'):
            import re
            match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if match:
                r, g, b = match.groups()
                return f'rgba({r},{g},{b},{alpha})'
        return f'rgba(100,100,100,{alpha})'

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Multiple Runs Novelty Search Dashboard",
                   style={'textAlign': 'center', 'marginBottom': 30}),

            # Status message
            html.Div(id='status-message', style={
                'textAlign': 'center',
                'marginBottom': 20,
                'padding': '10px',
                'backgroundColor': '#f0f0f0',
                'borderRadius': '5px',
                'display': 'none'
            }),

            # Generation slider
            html.Div([
                html.Label("Select Generation:",
                          style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Slider(
                    id='generation-slider',
                    min=0,
                    max=self.max_generation - 1,
                    step=1,
                    value=self.max_generation - 1,
                    marks={i: str(i) for i in range(0, self.max_generation,
                                                    max(1, self.max_generation // 10))},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Aggregated novelty comparison
            html.Div([
                html.H3("Novelty Score Comparison Across Runs"),
                dcc.Graph(id='novelty-comparison')
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Archive growth comparison
            html.Div([
                html.H3("Archive Growth Comparison"),
                dcc.Graph(id='archive-comparison')
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Robot Viewer (collapsible)
            html.Div([
                html.Div([
                    html.H3("Robot Viewer", style={'display': 'inline-block', 'margin': 0}),
                    html.Button(
                        "▶ Expand",
                        id='robot-viewer-collapse-btn',
                        style={
                            'float': 'right',
                            'background': 'none',
                            'border': '1px solid #ccc',
                            'padding': '5px 10px',
                            'cursor': 'pointer',
                            'borderRadius': '4px'
                        }
                    )
                ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                html.Div([
                    html.Div(id='robot-viewer-content', children=[
                        html.P("Click on any plot to view robots",
                              style={'textAlign': 'center', 'color': '#666',
                                    'fontSize': 14})
                    ])
                ], id='robot-viewer-container', style={'display': 'none'})
            ], style={
                'margin': '20px',
                'marginBottom': 40,
                'padding': '20px',
                'backgroundColor': '#f9f9f9',
                'borderRadius': '10px',
                'border': '2px solid #ddd'
            }),

            # Tabbed plots
            html.Div([
                dcc.Tabs(id='plot-tabs', value='space-coverage-tab', children=[
                    dcc.Tab(label='Combined Space Coverage',
                           value='space-coverage-tab'),
                    dcc.Tab(label='Feature Evolution Comparison',
                           value='evolution-tab'),
                    dcc.Tab(label='Diversity Comparison',
                           value='diversity-tab'),
                    dcc.Tab(label='Individual Runs',
                           value='individual-runs-tab'),
                ]),
                html.Div(id='tab-content')
            ], style={'margin': '20px'})
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [Output('robot-viewer-container', 'style'),
             Output('robot-viewer-collapse-btn', 'children')],
            Input('robot-viewer-collapse-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def toggle_robot_viewer(n_clicks):
            """Toggle robot viewer visibility."""
            if n_clicks is None:
                n_clicks = 0

            if n_clicks % 2 == 0:
                return {'display': 'none'}, "▶ Expand"
            else:
                return {'display': 'block'}, "▼ Collapse"

        @self.app.callback(
            Output('novelty-comparison', 'figure'),
            Input('generation-slider', 'value')
        )
        def update_novelty_comparison(selected_generation):
            """Update novelty comparison plot."""
            fig = go.Figure()

            generations = sorted(self.aggregated_novelty.keys())
            df_data = []

            for gen in generations:
                stats = self.aggregated_novelty[gen]
                df_data.append({
                    'generation': gen,
                    'mean_of_means': stats['mean_of_means'],
                    'std_of_means': stats['std_of_means'],
                    'mean_of_max': stats['mean_of_max'],
                    'std_of_max': stats['std_of_max']
                })

            df = pd.DataFrame(df_data)

            # Mean novelty across runs
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['mean_of_means'],
                mode='lines+markers',
                name='Mean Novelty',
                line=dict(color='blue', width=2)
            ))

            # Std band
            fig.add_trace(go.Scatter(
                x=df['generation'].tolist() + df['generation'][::-1].tolist(),
                y=(df['mean_of_means'] + df['std_of_means']).tolist() +
                  (df['mean_of_means'] - df['std_of_means'])[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='±1 Std (across runs)'
            ))

            # Max novelty across runs
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['mean_of_max'],
                mode='lines+markers',
                name='Mean Max Novelty',
                line=dict(color='green', width=2)
            ))

            # Highlight selected generation
            if selected_generation in self.aggregated_novelty:
                stats = self.aggregated_novelty[selected_generation]
                fig.add_trace(go.Scatter(
                    x=[selected_generation],
                    y=[stats['mean_of_means']],
                    mode='markers',
                    name=f'Gen {selected_generation}',
                    marker=dict(color='red', size=12, symbol='circle-open',
                              line=dict(width=3))
                ))

            fig.update_layout(
                title=f'Novelty Score Evolution (Aggregated across {self.num_runs} runs)',
                xaxis_title='Generation',
                yaxis_title='Novelty Score',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        @self.app.callback(
            Output('archive-comparison', 'figure'),
            Input('generation-slider', 'value')
        )
        def update_archive_comparison(selected_generation):
            """Update archive growth comparison."""
            fig = go.Figure()

            generations = sorted(self.archive_growth.keys())
            df_data = []

            for gen in generations:
                stats = self.archive_growth[gen]
                df_data.append({
                    'generation': gen,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                })

            df = pd.DataFrame(df_data)

            # Mean archive size
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['mean'],
                mode='lines+markers',
                name='Mean Archive Size',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.2)'
            ))

            # Min-max band
            fig.add_trace(go.Scatter(
                x=df['generation'].tolist() + df['generation'][::-1].tolist(),
                y=df['max'].tolist() + df['min'][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(128,0,128,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Min-Max Range'
            ))

            # Highlight selected generation
            if selected_generation in self.archive_growth:
                stats = self.archive_growth[selected_generation]
                fig.add_trace(go.Scatter(
                    x=[selected_generation],
                    y=[stats['mean']],
                    mode='markers',
                    name=f'Gen {selected_generation}',
                    marker=dict(color='red', size=12, symbol='circle-open',
                              line=dict(width=3))
                ))

            fig.update_layout(
                title=f'Archive Growth (Aggregated across {self.num_runs} runs)',
                xaxis_title='Generation',
                yaxis_title='Archive Size',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('plot-tabs', 'value'),
             Input('generation-slider', 'value')]
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            if active_tab == 'space-coverage-tab':
                return self._create_combined_space_coverage()
            elif active_tab == 'evolution-tab':
                return self._create_feature_evolution_comparison()
            elif active_tab == 'diversity-tab':
                return self._create_diversity_comparison(selected_generation)
            elif active_tab == 'individual-runs-tab':
                return self._create_individual_runs_view(selected_generation)

            return html.Div("Select a tab to view plots")

    def _create_combined_space_coverage(self):
        """Create combined space coverage plot for all runs."""
        # Combine all descriptors
        all_descriptors = np.vstack(self.run_descriptors)

        # PCA projection
        if len(all_descriptors) > 2:
            pca = PCA(n_components=2, random_state=42)
            projection = pca.fit_transform(all_descriptors)

            fig = go.Figure()

            # Add each run with different color
            offset = 0
            for run_idx, descriptors in enumerate(self.run_descriptors):
                run_projection = projection[offset:offset + len(descriptors)]
                generations = self.run_generations[run_idx]

                fig.add_trace(go.Scatter(
                    x=run_projection[:, 0],
                    y=run_projection[:, 1],
                    mode='markers',
                    name=self.run_names[run_idx],
                    marker=dict(
                        size=5,
                        color=self.run_colors[run_idx],
                        opacity=0.6
                    ),
                    text=[f"{self.run_names[run_idx]}<br>Gen: {g}"
                          for g in generations],
                    hoverinfo='text'
                ))

                offset += len(descriptors)

            fig.update_layout(
                title=f'Combined Morphological Space Coverage ({self.num_runs} runs)',
                xaxis_title='PC1',
                yaxis_title='PC2',
                height=700,
                hovermode='closest'
            )

            return dcc.Graph(figure=fig)

        return html.Div("Not enough data for projection")

    def _create_feature_evolution_comparison(self):
        """Create feature evolution comparison across runs."""
        fig = go.Figure()

        for run_idx, (descriptors, generations) in enumerate(
            zip(self.run_descriptors, self.run_generations)
        ):
            # Compute mean descriptor per generation
            max_gen = int(generations.max()) + 1
            gen_means = []

            for gen in range(max_gen):
                gen_mask = generations == gen
                if gen_mask.any():
                    gen_desc = descriptors[gen_mask]
                    # Average across all features
                    mean_val = np.mean(gen_desc)
                    gen_means.append(mean_val)
                else:
                    gen_means.append(np.nan)

            color = self.run_colors[run_idx]

            fig.add_trace(go.Scatter(
                x=list(range(max_gen)),
                y=gen_means,
                mode='lines',
                name=self.run_names[run_idx],
                line=dict(color=color, width=2),
                opacity=0.7
            ))

        fig.update_layout(
            title='Average Feature Value Evolution Across Runs',
            xaxis_title='Generation',
            yaxis_title='Mean Feature Value',
            height=500,
            hovermode='x unified'
        )

        return dcc.Graph(figure=fig)

    def _create_diversity_comparison(self, selected_generation):
        """Create diversity comparison radar chart."""
        fig = go.Figure()

        for run_idx, (descriptors, generations) in enumerate(
            zip(self.run_descriptors, self.run_generations)
        ):
            # Get descriptors for selected generation
            gen_mask = generations == selected_generation
            if gen_mask.any():
                gen_descriptors = descriptors[gen_mask]

                # Compute std dev for each feature
                diversities = [np.std(gen_descriptors[:, i])
                             for i in range(len(self.feature_names))]

                color = self.run_colors[run_idx]

                fig.add_trace(go.Scatterpolar(
                    r=diversities,
                    theta=self.feature_names,
                    fill='toself',
                    name=self.run_names[run_idx],
                    line_color=color,
                    fillcolor=self._color_to_rgba(color, 0.2)
                ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            title=f'Morphological Diversity Comparison - Generation {selected_generation}',
            height=600,
            showlegend=True
        )

        return dcc.Graph(figure=fig)

    def _create_individual_runs_view(self, selected_generation):
        """Create individual runs view with separate plots."""
        from plotly.subplots import make_subplots

        rows = (self.num_runs + 1) // 2
        cols = min(2, self.num_runs)

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=self.run_names,
            specs=[[{'type': 'scatter'}] * cols for _ in range(rows)]
        )

        for run_idx, (descriptors, generations) in enumerate(
            zip(self.run_descriptors, self.run_generations)
        ):
            row = run_idx // 2 + 1
            col = run_idx % 2 + 1

            # Get descriptors for selected generation
            gen_mask = generations == selected_generation
            if gen_mask.any():
                gen_descriptors = descriptors[gen_mask]

                # Use first two features for scatter
                fig.add_trace(
                    go.Scatter(
                        x=gen_descriptors[:, 0],
                        y=gen_descriptors[:, 1],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=self.run_colors[run_idx],
                            opacity=0.7
                        ),
                        showlegend=False,
                        name=self.run_names[run_idx]
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title=f'Individual Runs - Generation {selected_generation}',
            height=400 * rows,
            showlegend=False
        )

        # Update axis labels
        for i in range(1, self.num_runs + 1):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            fig.update_xaxes(title_text='Branching', row=row, col=col)
            fig.update_yaxes(title_text='Limbs', row=row, col=col)

        return dcc.Graph(figure=fig)

    def run(self, host='127.0.0.1', port=8053, debug=True):
        """Run the dashboard server."""
        print(f"Starting Multiple Runs Novelty Dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def load_config_from_saved_json(config_path: Path) -> Any:
    """Load configuration from saved JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    resolved = config_data["resolved_settings"]
    genotype_name = resolved["genotype_name"]
    genotype = GENOTYPES_MAPPING[genotype_name]

    class DashboardConfig:
        def __init__(self):
            self.genotype = genotype
            self.task = resolved["task"]
            self.task_params = resolved["task_params"]
            self.genotype_name = genotype_name

    return DashboardConfig()


def load_populations_from_database(db_path: Path) -> Tuple[List[Population], Any]:
    """Load populations from database."""
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
        raise FileNotFoundError(f"No configuration file found: {json_config_path}")

    return populations, config


def main():
    """Main entry point for multiple runs novelty dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multiple runs novelty search dashboard"
    )
    parser.add_argument("--db_paths", nargs='+', required=True,
                       help="Paths to database files for each run")
    parser.add_argument("--names", nargs='+',
                       help="Names for each run (optional)")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8053, help="Dashboard port")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    # Load all runs
    runs_data = []
    for db_path in args.db_paths:
        try:
            populations, config = load_populations_from_database(Path(db_path))
            runs_data.append((populations, config))
        except Exception as e:
            console.log(f"Error loading {db_path}: {e}")
            continue

    if not runs_data:
        console.log("No valid databases loaded!")
        return

    # Create dashboard
    run_names = args.names if args.names else None
    dashboard = MultipleRunsNoveltyDashboard(runs_data, run_names)
    dashboard.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()