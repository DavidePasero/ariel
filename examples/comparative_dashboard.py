#!/usr/bin/env python3
"""
Comparative Evolution Dashboard

This module provides an interactive web dashboard for comparing evolutionary
computation results across different genotypes using Plotly Dash. It loads
data from multiple SQLite databases and displays comparative analysis.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
import tomllib
import json
from typing import List, Callable, Any, Dict, Tuple
from rich.console import Console
from sqlmodel import Session, create_engine, select

from plotly_morphology_analysis import PlotlyMorphologyAnalyzer
from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING

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
            if hasattr(config, 'task_params') and 'target_robot_path' in config.task_params:
                analyzer.load_target_robots(str(config.task_params["target_robot_path"]))
            self.analyzers[genotype_name] = analyzer

        # Pre-compute fitness timelines for all genotypes
        self._compute_fitness_timelines()

        # Cache for computed descriptors per generation per genotype
        self._descriptor_cache = {}

        # Morphological feature names
        self.feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        # Color mapping for genotypes
        self.colors = {
            self.genotype_names[0]: '#1f77b4',  # Blue
            self.genotype_names[1]: '#ff7f0e' if len(self.genotype_names) > 1 else '#1f77b4',  # Orange
            self.genotype_names[2] if len(self.genotype_names) > 2 else "": '#2ca02c' if len(self.genotype_names) > 2 else '#1f77b4'   # Green
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
                        'generation': gen_idx,
                        'avg_fitness': np.mean(fitnesses),
                        'std_fitness': np.std(fitnesses),
                        'best_fitness': max(fitnesses),
                        'worst_fitness': min(fitnesses),
                        'median_fitness': np.median(fitnesses)
                    })

            self.fitness_timelines[genotype_name] = timeline

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
            'descriptors': analyzer.descriptors.copy(),
            'fitness_scores': analyzer.fitness_scores.copy()
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
            html.H1("Comparative Evolution Dashboard",
                   style={'textAlign': 'center', 'marginBottom': 30}),

            # Generation control section
            html.Div([
                html.Label("Select Generation:",
                          style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Slider(
                    id='generation-slider',
                    min=0,
                    max=max_generation,
                    step=1,
                    value=max_generation,
                    marks={i: str(i) for i in range(0, max_generation + 1, max(1, max_generation // 10))},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Fitness comparison (collapsible)
            html.Div([
                html.Div([
                    html.H3("Fitness Comparison Across Genotypes", style={'display': 'inline-block', 'margin': 0}),
                    html.Button(
                        "▼ Collapse",
                        id='fitness-collapse-btn',
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
                    dcc.Graph(id='fitness-comparison')
                ], id='fitness-plot-container', style={'display': 'block'})
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Tabbed plots section
            html.Div([
                dcc.Tabs(id='plot-tabs', value='single-feature-tab', children=[
                    dcc.Tab(label='Single Feature Evolution', value='single-feature-tab'),
                    dcc.Tab(label='Fitness Distributions', value='distribution-tab'),
                    dcc.Tab(label='Morphological Diversity', value='diversity-tab'),
                ]),
                html.Div(id='tab-content')
            ], style={'margin': '20px'})
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [Output('fitness-plot-container', 'style'),
             Output('fitness-collapse-btn', 'children')],
            Input('fitness-collapse-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def toggle_fitness_plot(n_clicks):
            """Toggle fitness plot visibility."""
            if n_clicks is None:
                n_clicks = 0

            if n_clicks % 2 == 0:
                # Show plot
                return {'display': 'block'}, "▼ Collapse"
            else:
                # Hide plot
                return {'display': 'none'}, "▶ Expand"

        @self.app.callback(
            Output('fitness-comparison', 'figure'),
            Input('generation-slider', 'value')
        )
        def update_fitness_comparison(selected_generation):
            """Update fitness comparison plot with highlighted generation."""
            fig = go.Figure()

            for genotype_name in self.genotype_names:
                timeline = self.fitness_timelines[genotype_name]
                if not timeline:
                    continue

                df = pd.DataFrame(timeline)
                color = self.colors.get(genotype_name, '#1f77b4')

                # Add mean fitness line
                fig.add_trace(go.Scatter(
                    x=df['generation'],
                    y=df['avg_fitness'],
                    mode='lines+markers',
                    name=f'{genotype_name} - Mean',
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                ))

                # Add std deviation band
                fig.add_trace(go.Scatter(
                    x=df['generation'].tolist() + df['generation'][::-1].tolist(),
                    y=(df['avg_fitness'] + df['std_fitness']).tolist() +
                      (df['avg_fitness'] - df['std_fitness'])[::-1].tolist(),
                    fill='tonext',
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f'{genotype_name} - ±1 STD'
                ))

                # Highlight selected generation
                if selected_generation < len(df):
                    selected_row = df.iloc[selected_generation]
                    fig.add_trace(go.Scatter(
                        x=[selected_row['generation']],
                        y=[selected_row['avg_fitness']],
                        mode='markers',
                        name=f'{genotype_name} - Gen {selected_generation}',
                        marker=dict(color=color, size=12, symbol='circle-open',
                                  line=dict(width=3)),
                        showlegend=False
                    ))

            fig.update_layout(
                title='Mean Fitness Evolution with Standard Deviation',
                xaxis_title='Generation',
                yaxis_title='Fitness',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        @self.app.callback(
            Output('single-feature-graph', 'figure'),
            Input('feature-dropdown', 'value'),
            prevent_initial_call=True
        )
        def update_single_feature_plot(feature_idx):
            """Update single feature plot based on selected feature."""
            if feature_idx is None:
                return go.Figure()
            return self._plot_feature_evolution_comparison(feature_idx)

        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('plot-tabs', 'value'),
             Input('generation-slider', 'value')]
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            if active_tab == 'distribution-tab':
                return self._create_distribution_comparison(selected_generation)
            elif active_tab == 'diversity-tab':
                return self._create_diversity_comparison(selected_generation)
            elif active_tab == 'single-feature-tab':
                return self._create_single_feature_plot()

            return html.Div("Select a tab to view plots")

    def _create_single_feature_plot(self):
        """Create single feature evolution comparison with feature selector."""
        return html.Div([
            html.Div([
                html.Label("Select Feature:",
                          style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': feature, 'value': i}
                           for i, feature in enumerate(self.feature_names)],
                    value=0,
                    style={'marginBottom': 20}
                )
            ], style={'width': '300px', 'margin': '0 auto'}),
            dcc.Graph(id='single-feature-graph')
        ])

    def _plot_feature_evolution_comparison(self, feature_idx: int):
        """Plot evolution of a single morphological feature across genotypes."""
        if feature_idx is None:
            return go.Figure()

        feature_name = self.feature_names[feature_idx]
        fig = go.Figure()

        for genotype_name in self.genotype_names:
            populations, config = self.genotype_data[genotype_name]
            color = self.colors.get(genotype_name, '#1f77b4')

            # Compute feature values for all generations
            generation_data = []

            for gen_idx in range(len(populations)):
                gen_data = self._get_generation_data(genotype_name, gen_idx)
                if gen_data['descriptors'].size > 0:
                    feature_values = gen_data['descriptors'][:, feature_idx]

                    generation_data.append({
                        'generation': gen_idx,
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values)
                    })

            if not generation_data:
                continue

            df = pd.DataFrame(generation_data)

            # Add mean line
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['mean'],
                mode='lines+markers',
                name=f'{genotype_name} - Mean',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))

            # Add standard deviation bands
            fig.add_trace(go.Scatter(
                x=df['generation'].tolist() + df['generation'][::-1].tolist(),
                y=(df['mean'] + df['std']).tolist() + (df['mean'] - df['std'])[::-1].tolist(),
                fill='tonext',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f'{genotype_name} - ±1 STD'
            ))

        # Add target robot values as horizontal lines
        target_added = False
        for genotype_name in self.genotype_names:
            analyzer = self.analyzers[genotype_name]
            if hasattr(analyzer, 'target_descriptors') and len(analyzer.target_descriptors) > 0:
                # Since there's only one target robot per genotype, use the first one
                target_value = analyzer.target_descriptors[0, feature_idx]
                target_name = analyzer.target_names[0] if analyzer.target_names else "Target"

                # Only add one target line (they should be the same across genotypes)
                if not target_added:
                    fig.add_hline(
                        y=target_value,
                        line_dash="solid",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"Target: {target_name} ({target_value:.2f})",
                        annotation_position="top right"
                    )
                    target_added = True
                break  # Only need one target since they should be the same

        fig.update_layout(
            title=f'{feature_name} Evolution Comparison',
            xaxis_title='Generation',
            yaxis_title=f'{feature_name} Value',
            height=500,
            showlegend=True,
            hovermode='x unified'
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
            color = self.colors.get(genotype_name, '#1f77b4')

            fig.add_trace(go.Histogram(
                x=fitnesses,
                name=genotype_name,
                opacity=0.7,
                nbinsx=20,
                marker_color=color
            ))

        fig.update_layout(
            title=f'Fitness Distributions - Generation {generation}',
            xaxis_title='Fitness',
            yaxis_title='Count',
            height=500,
            barmode='overlay',
            showlegend=True
        )

        return dcc.Graph(figure=fig)

    def _create_diversity_comparison(self, generation: int):
        """Create morphological diversity comparison for selected generation."""
        fig = go.Figure()

        for genotype_name in self.genotype_names:
            gen_data = self._get_generation_data(genotype_name, generation)

            if gen_data['descriptors'].size == 0:
                continue

            descriptors = gen_data['descriptors']
            color = self.colors.get(genotype_name, '#1f77b4')

            # Compute diversity metrics for each feature
            diversity_data = []
            for i, feature_name in enumerate(self.feature_names):
                feature_values = descriptors[:, i]
                diversity = np.std(feature_values)  # Simple diversity measure
                diversity_data.append(diversity)

            fig.add_trace(go.Scatterpolar(
                r=diversity_data,
                theta=self.feature_names,
                fill='toself',
                name=genotype_name,
                line_color=color,
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.3])}'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max([
                    max(self._get_diversity_for_genotype(gname, generation))
                    for gname in self.genotype_names
                    if self._get_diversity_for_genotype(gname, generation)
                ]) if any(self._get_diversity_for_genotype(gname, generation)
                         for gname in self.genotype_names) else 1])
            ),
            title=f'Morphological Diversity Comparison - Generation {generation}',
            height=500,
            showlegend=True
        )

        return dcc.Graph(figure=fig)

    def _get_diversity_for_genotype(self, genotype_name: str, generation: int) -> List[float]:
        """Helper to get diversity values for a genotype at a generation."""
        try:
            gen_data = self._get_generation_data(genotype_name, generation)
            if gen_data['descriptors'].size == 0:
                return []

            descriptors = gen_data['descriptors']
            return [np.std(descriptors[:, i]) for i in range(len(self.feature_names))]
        except:
            return []

    def run(self, host='127.0.0.1', port=8051, debug=True):
        """Run the dashboard server."""
        print(f"Starting Comparative Evolution Dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


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
            self.db_file_path = Path(cfg["data"]["output_folder"]) / cfg["data"]["db_file_name"]

    return DashboardConfig()


def load_populations_from_database(db_path: Path, genotype_name: str) -> Tuple[List[Population], Any]:
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

            population = [Individual.model_validate(ind.model_dump()) for ind in individuals]
            populations.append(population)

    # Load config - first try the saved JSON config file
    config_filename = db_path.stem + "_config.json"
    json_config_path = db_path.parent / config_filename

    if json_config_path.exists():
        console.log(f"Loading saved configuration from: {json_config_path}")
        config = load_config_from_saved_json(json_config_path)
    else:
        # Fallback to TOML config files
        console.log(f"No saved config found at {json_config_path}, trying TOML fallback")
        config_path = db_path.parent / "config.toml"
        if not config_path.exists():
            # Try examples/config.toml as fallback
            config_path = Path("examples/config.toml")

        if config_path.exists():
            console.log(f"Loading TOML configuration from: {config_path}")
            config = load_config_for_genotype(config_path)
        else:
            raise FileNotFoundError(f"No configuration file found for database {db_path}")

    return populations, config


def main():
    """Main entry point for comparative dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Comparative evolution dashboard")
    parser.add_argument("--db_paths", nargs='+', help="Paths to database files",
                        default=["__data__/tree.db", "__data__/lsystem.db"], required=False)
    parser.add_argument("--names", nargs='+', help="Names for genotypes (default: genotype names from config)",
                        default=["Tree", "L-System"], required=False)
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8051, help="Dashboard port")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    if len(args.db_paths) > 3:
        console.log("Warning: Only first 3 databases will be used for comparison")
        args.db_paths = args.db_paths[:3]

    genotype_data = {}

    for i, db_path in enumerate(args.db_paths):
        db_path = Path(db_path)

        try:
            populations, config = load_populations_from_database(db_path, f"genotype_{i}")

            if args.names and i < len(args.names):
                genotype_name = args.names[i]
            else:
                # Use the genotype name from the saved config if available
                if hasattr(config, 'genotype_name'):
                    genotype_name = config.genotype_name.capitalize()
                else:
                    genotype_name = config.genotype.__name__.replace('Genotype', '')

            genotype_data[genotype_name] = (populations, config)
            console.log(f"Loaded {genotype_name}: {len(populations)} generations")

            # Log additional config info if available
            if hasattr(config, 'mutation_name') and hasattr(config, 'crossover_name'):
                console.log(f"  - Mutation: {config.mutation_name}, Crossover: {config.crossover_name}")
            if hasattr(config, 'task'):
                console.log(f"  - Task: {config.task}")

        except Exception as e:
            console.log(f"Error loading {db_path}: {e}")
            continue

    if not genotype_data:
        console.log("No valid databases loaded!")
        return

    console.log(f"Starting comparative dashboard with {len(genotype_data)} genotypes")
    dashboard = ComparativeEvolutionDashboard(genotype_data)
    dashboard.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()
