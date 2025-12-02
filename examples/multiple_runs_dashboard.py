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
from typing import List, Dict, Tuple, Any
from rich.console import Console
from sqlmodel import Session, create_engine, select
from networkx.readwrite import json_graph

from plotly_morphology_analysis import PlotlyMorphologyAnalyzer
from ariel.ec.a001 import Individual
from ariel.ec.genotypes.genotype_mapping import GENOTYPES_MAPPING

console = Console()
type Population = List[Individual]


class MultipleRunsDashboard:
    """Interactive dashboard for analyzing multiple evolutionary runs."""

    def __init__(self, runs_data: List[Tuple[List[Population], Any]], run_names: List[str] = None):
        """Initialize dashboard with multiple run data.

        Args:
            runs_data: List of (populations, config) tuples, one per run
            run_names: Optional names for each run (default: "Run 1", "Run 2", ...)
        """
        self.runs_data = runs_data
        self.num_runs = len(runs_data)

        if run_names is None:
            self.run_names = [f"Run {i+1}" for i in range(self.num_runs)]
        else:
            self.run_names = run_names

        # Use the first run's config as the reference
        _, self.config = runs_data[0]

        # Initialize analyzer (shared for all runs, same genotype)
        self.analyzer = PlotlyMorphologyAnalyzer()

        # Load target robots if available
        if hasattr(self.config, 'task_params') and 'target_robot_path' in self.config.task_params:
            self.analyzer.load_target_robots(str(self.config.task_params["target_robot_path"]))

        # Pre-compute aggregated fitness statistics
        self._compute_aggregated_fitness()

        # Cache for computed descriptors per run per generation
        self._descriptor_cache = {}

        # Cache for individual data (for click handling)
        self._individual_cache = {}

        # Morphological feature names
        self.feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        # Create dl_robots directory if it doesn't exist
        self.dl_robots_path = Path("examples/dl_robots")
        self.dl_robots_path.mkdir(exist_ok=True)

        # Color scheme for individual runs
        self.run_colors = px.colors.qualitative.Set2[:self.num_runs]

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
        if color.startswith('#'):
            # Hex color
            rgb = px.colors.hex_to_rgb(color)
            return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'
        elif color.startswith('rgb'):
            # Already in rgb/rgba format
            # Extract RGB values
            import re
            match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if match:
                r, g, b = match.groups()
                return f'rgba({r},{g},{b},{alpha})'
        # Fallback
        return f'rgba(100,100,100,{alpha})'

    def _compute_aggregated_fitness(self):
        """Compute aggregated fitness statistics across all runs."""
        # Find max number of generations across all runs
        self.max_generation = max(len(populations) for populations, _ in self.runs_data)

        # Store individual run timelines for overlay plots
        self.individual_run_timelines = []

        for run_idx, (populations, config) in enumerate(self.runs_data):
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
                    avg_fitnesses.append(timeline[gen_idx]['avg_fitness'])
                    best_fitnesses.append(timeline[gen_idx]['best_fitness'])
                    worst_fitnesses.append(timeline[gen_idx]['worst_fitness'])
                    median_fitnesses.append(timeline[gen_idx]['median_fitness'])

            if avg_fitnesses:
                self.aggregated_timeline.append({
                    'generation': gen_idx,
                    'mean_avg_fitness': np.mean(avg_fitnesses),
                    'std_avg_fitness': np.std(avg_fitnesses),
                    'mean_best_fitness': np.mean(best_fitnesses),
                    'std_best_fitness': np.std(best_fitnesses),
                    'mean_worst_fitness': np.mean(worst_fitnesses),
                    'std_worst_fitness': np.std(worst_fitnesses),
                    'mean_median_fitness': np.mean(median_fitnesses),
                    'overall_best': max(best_fitnesses),
                    'overall_worst': min(worst_fitnesses)
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
            'descriptors': self.analyzer.descriptors.copy(),
            'fitness_scores': self.analyzer.fitness_scores.copy()
        }

        # Cache individual data for click handling
        self._individual_cache[cache_key] = {
            'population': population.copy(),
            'decoder': decoder,
            'run_idx': run_idx
        }

        return self._descriptor_cache[cache_key]

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Multiple Runs Analysis Dashboard",
                   style={'textAlign': 'center', 'marginBottom': 10}),

            html.Div([
                html.P(f"Analyzing {self.num_runs} independent evolutionary runs",
                      style={'textAlign': 'center', 'color': '#666', 'marginBottom': 30})
            ]),

            # Status message area
            html.Div(id='status-message', style={
                'textAlign': 'center',
                'marginBottom': 20,
                'padding': '10px',
                'backgroundColor': '#f0f0f0',
                'borderRadius': '5px',
                'display': 'none'
            }),

            # Generation control section
            html.Div([
                html.Label("Select Generation:", style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Slider(
                    id='generation-slider',
                    min=0,
                    max=self.max_generation - 1,
                    step=1,
                    value=self.max_generation - 1,
                    marks={i: str(i) for i in range(0, self.max_generation, max(1, self.max_generation // 10))},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Aggregated fitness plot (always visible)
            html.Div([
                html.H3("Aggregated Fitness Across Runs"),
                html.P("Mean ± standard deviation across all runs",
                      style={'color': '#666', 'fontSize': '14px'}),
                dcc.Graph(id='aggregated-fitness')
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Individual runs plot (collapsible)
            html.Div([
                html.Div([
                    html.H3("Individual Run Trajectories", style={'display': 'inline-block', 'margin': 0}),
                    html.Button(
                        "▼ Collapse",
                        id='runs-collapse-btn',
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
                    dcc.Graph(id='individual-runs')
                ], id='runs-plot-container', style={'display': 'block'})
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Tabbed plots section
            html.Div([
                dcc.Tabs(id='plot-tabs', value='feature-evolution-tab', children=[
                    dcc.Tab(label='Feature Evolution', value='feature-evolution-tab'),
                    dcc.Tab(label='Fitness Distributions', value='distribution-tab'),
                    dcc.Tab(label='Diversity Metrics', value='diversity-tab'),
                    dcc.Tab(label='Run Comparison', value='comparison-tab'),
                ]),
                html.Div(id='tab-content')
            ], style={'margin': '20px'})
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [Output('runs-plot-container', 'style'),
             Output('runs-collapse-btn', 'children')],
            Input('runs-collapse-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def toggle_runs_plot(n_clicks):
            """Toggle individual runs plot visibility."""
            if n_clicks is None:
                n_clicks = 0

            if n_clicks % 2 == 0:
                return {'display': 'block'}, "▼ Collapse"
            else:
                return {'display': 'none'}, "▶ Expand"

        @self.app.callback(
            Output('aggregated-fitness', 'figure'),
            Input('generation-slider', 'value')
        )
        def update_aggregated_fitness(selected_generation):
            """Update aggregated fitness plot."""
            if not self.aggregated_timeline:
                return go.Figure()

            df = pd.DataFrame(self.aggregated_timeline)
            fig = go.Figure()

            # Add mean of average fitness across runs
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['mean_avg_fitness'],
                mode='lines+markers',
                name='Mean Avg Fitness',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))

            # Add std deviation band for average fitness
            fig.add_trace(go.Scatter(
                x=df['generation'].tolist() + df['generation'][::-1].tolist(),
                y=(df['mean_avg_fitness'] + df['std_avg_fitness']).tolist() +
                  (df['mean_avg_fitness'] - df['std_avg_fitness'])[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='±1 STD (Avg Fitness)'
            ))

            # Add mean of best fitness across runs
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['mean_best_fitness'],
                mode='lines+markers',
                name='Mean Best Fitness',
                line=dict(color='green', width=3),
                marker=dict(size=6)
            ))

            # Add std deviation band for best fitness
            fig.add_trace(go.Scatter(
                x=df['generation'].tolist() + df['generation'][::-1].tolist(),
                y=(df['mean_best_fitness'] + df['std_best_fitness']).tolist() +
                  (df['mean_best_fitness'] - df['std_best_fitness'])[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,200,100,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='±1 STD (Best Fitness)'
            ))

            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(go.Scatter(
                    x=[selected_row['generation']],
                    y=[selected_row['mean_avg_fitness']],
                    mode='markers',
                    name=f'Generation {selected_generation}',
                    marker=dict(color='red', size=15, symbol='circle-open',
                               line=dict(width=3)),
                    showlegend=False
                ))

            fig.update_layout(
                title='Mean Fitness Evolution Across Runs',
                xaxis_title='Generation',
                yaxis_title='Fitness',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        @self.app.callback(
            Output('individual-runs', 'figure'),
            Input('generation-slider', 'value')
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
                fig.add_trace(go.Scatter(
                    x=df['generation'],
                    y=df['avg_fitness'],
                    mode='lines+markers',
                    name=f'{self.run_names[run_idx]} - Avg',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    opacity=0.7
                ))

                # Add best fitness line for this run
                fig.add_trace(go.Scatter(
                    x=df['generation'],
                    y=df['best_fitness'],
                    mode='lines',
                    name=f'{self.run_names[run_idx]} - Best',
                    line=dict(color=color, width=1, dash='dot'),
                    showlegend=False,
                    opacity=0.5
                ))

            fig.update_layout(
                title='Individual Run Fitness Trajectories',
                xaxis_title='Generation',
                yaxis_title='Fitness',
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
            if active_tab == 'feature-evolution-tab':
                return self._create_feature_evolution_plot()
            elif active_tab == 'distribution-tab':
                return self._create_distribution_plot(selected_generation)
            elif active_tab == 'diversity-tab':
                return self._create_diversity_plot(selected_generation)
            elif active_tab == 'comparison-tab':
                return self._create_comparison_plot(selected_generation)

            return html.Div("Select a tab to view plots")

        @self.app.callback(
            Output('feature-evolution-graph', 'figure'),
            Input('feature-dropdown', 'value'),
            prevent_initial_call=True
        )
        def update_feature_plot(feature_idx):
            """Update feature evolution plot."""
            if feature_idx is None:
                return go.Figure()
            return self._plot_feature_evolution(feature_idx)

        @self.app.callback(
            [Output('status-message', 'children'),
             Output('status-message', 'style')],
            Input('comparison-scatter', 'clickData'),
            State('generation-slider', 'value'),
            prevent_initial_call=True
        )
        def handle_click(clickData, generation):
            """Handle click on scatter plot to save robot."""
            if not clickData or 'points' not in clickData:
                return "", {'display': 'none'}

            try:
                point = clickData['points'][0]
                point_index = point.get('pointIndex', point.get('pointNumber', 0))

                if not hasattr(self, '_click_data') or point_index not in self._click_data:
                    return "Error: Click data not found", {
                        'display': 'block',
                        'backgroundColor': '#ffcccc',
                        'color': 'red',
                        'textAlign': 'center',
                        'marginBottom': 20,
                        'padding': '10px',
                        'borderRadius': '5px'
                    }

                click_info = self._click_data[point_index]
                run_idx = click_info['run_idx']
                individual_idx = click_info['individual_idx']

                cache_key = f"run{run_idx}_gen{generation}"
                if cache_key not in self._individual_cache:
                    return "Error: Individual data not found", {
                        'display': 'block',
                        'backgroundColor': '#ffcccc',
                        'color': 'red',
                        'textAlign': 'center',
                        'marginBottom': 20,
                        'padding': '10px',
                        'borderRadius': '5px'
                    }

                individual_data = self._individual_cache[cache_key]
                population = individual_data['population']
                decoder = individual_data['decoder']

                if individual_idx >= len(population):
                    return f"Error: Individual index out of range", {
                        'display': 'block',
                        'backgroundColor': '#ffcccc',
                        'color': 'red',
                        'textAlign': 'center',
                        'marginBottom': 20,
                        'padding': '10px',
                        'borderRadius': '5px'
                    }

                individual = population[individual_idx]
                robot_graph = decoder(individual)
                robot_data = json_graph.node_link_data(robot_graph, edges="edges")

                filename = f"robot_run{run_idx}_gen{generation}_ind{individual_idx}_fit{individual.fitness:.3f}.json"
                filepath = self.dl_robots_path / filename

                with open(filepath, 'w') as f:
                    json.dump(robot_data, f, indent=2)

                return f"Robot saved: {filename}", {
                    'display': 'block',
                    'backgroundColor': '#ccffcc',
                    'color': 'green',
                    'textAlign': 'center',
                    'marginBottom': 20,
                    'padding': '10px',
                    'borderRadius': '5px'
                }

            except Exception as e:
                return f"Error saving robot: {str(e)}", {
                    'display': 'block',
                    'backgroundColor': '#ffcccc',
                    'color': 'red',
                    'textAlign': 'center',
                    'marginBottom': 20,
                    'padding': '10px',
                    'borderRadius': '5px'
                }

    def _create_feature_evolution_plot(self):
        """Create feature evolution plot with feature selector."""
        return html.Div([
            html.Div([
                html.Label("Select Morphological Feature:",
                          style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': feature, 'value': i}
                            for i, feature in enumerate(self.feature_names)],
                    value=0,
                    style={'marginBottom': 20}
                )
            ], style={'width': '300px', 'margin': '0 auto'}),
            dcc.Graph(id='feature-evolution-graph')
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
                    if gen_data['descriptors'].size > 0:
                        feature_values = gen_data['descriptors'][:, feature_idx]
                        feature_means.append(np.mean(feature_values))
                        feature_stds.append(np.std(feature_values))

            if feature_means:
                aggregated_data.append({
                    'generation': gen_idx,
                    'mean_of_means': np.mean(feature_means),
                    'std_of_means': np.std(feature_means),
                    'mean_diversity': np.mean(feature_stds)
                })

        if not aggregated_data:
            return go.Figure()

        df = pd.DataFrame(aggregated_data)
        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=df['generation'],
            y=df['mean_of_means'],
            mode='lines+markers',
            name='Mean Across Runs',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))

        # Add std deviation band
        fig.add_trace(go.Scatter(
            x=df['generation'].tolist() + df['generation'][::-1].tolist(),
            y=(df['mean_of_means'] + df['std_of_means']).tolist() +
              (df['mean_of_means'] - df['std_of_means'])[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='±1 STD Across Runs'
        ))

        # Add mean within-run diversity
        fig.add_trace(go.Scatter(
            x=df['generation'],
            y=df['mean_diversity'],
            mode='lines+markers',
            name='Mean Within-Run Diversity',
            line=dict(color='purple', width=2, dash='dot'),
            marker=dict(size=5)
        ))

        # Add target robot values if available
        if hasattr(self.analyzer, 'target_descriptors') and len(self.analyzer.target_descriptors) > 0:
            for i, (target_desc, target_name) in enumerate(zip(self.analyzer.target_descriptors,
                                                                self.analyzer.target_names)):
                target_value = target_desc[feature_idx]
                fig.add_hline(
                    y=target_value,
                    line_dash="solid",
                    line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                    line_width=2,
                    annotation_text=f"Target: {target_name} ({target_value:.2f})",
                    annotation_position="top right"
                )

        fig.update_layout(
            title=f'{feature_name} Evolution Across Multiple Runs',
            xaxis_title='Generation',
            yaxis_title=f'{feature_name} Value',
            height=500,
            showlegend=True,
            hovermode='x unified'
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

            fig.add_trace(go.Histogram(
                x=fitnesses,
                name=self.run_names[run_idx],
                opacity=0.6,
                nbinsx=20,
                marker_color=color
            ))

        fig.update_layout(
            title=f'Fitness Distributions Across Runs - Generation {generation}',
            xaxis_title='Fitness',
            yaxis_title='Count',
            height=500,
            barmode='overlay',
            showlegend=True
        )

        return dcc.Graph(figure=fig)

    def _create_diversity_plot(self, generation: int):
        """Create morphological diversity comparison across runs."""
        fig = go.Figure()

        for run_idx in range(self.num_runs):
            gen_data = self._get_generation_data(run_idx, generation)

            if gen_data['descriptors'].size == 0:
                continue

            descriptors = gen_data['descriptors']
            color = self.run_colors[run_idx % len(self.run_colors)]

            # Compute diversity (std) for each feature
            diversity_data = []
            for i in range(len(self.feature_names)):
                feature_values = descriptors[:, i]
                diversity = np.std(feature_values)
                diversity_data.append(diversity)

            fig.add_trace(go.Scatterpolar(
                r=diversity_data,
                theta=self.feature_names,
                fill='toself',
                name=self.run_names[run_idx],
                line_color=color,
                fillcolor=self._color_to_rgba(color, alpha=0.3)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, None])
            ),
            title=f'Morphological Diversity Across Runs - Generation {generation}',
            height=600,
            showlegend=True
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

            if gen_data['descriptors'].size == 0:
                continue

            descriptors = gen_data['descriptors']
            fitness_scores = gen_data['fitness_scores']
            color = self.run_colors[run_idx % len(self.run_colors)]

            # Use first two features for scatter
            x_values = descriptors[:, 0]  # Branching
            y_values = descriptors[:, 1]  # Limbs

            # Store click data mapping
            for i in range(len(x_values)):
                self._click_data[current_index] = {
                    'run_idx': run_idx,
                    'individual_idx': i,
                    'generation': generation
                }
                current_index += 1

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                name=self.run_names[run_idx],
                marker=dict(
                    color=fitness_scores[0] if len(fitness_scores) > 0 else [0] * len(x_values),
                    colorscale='Viridis',
                    size=8,
                    opacity=0.6,
                    line=dict(width=0.5, color='white'),
                    colorbar=dict(title='Fitness') if run_idx == 0 else None
                ),
                hovertemplate=f'<b>{self.run_names[run_idx]}</b><br>' +
                             'Branching: %{x:.3f}<br>' +
                             'Limbs: %{y:.3f}<br>' +
                             'Fitness: %{marker.color:.3f}<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title=f'Morphological Space Comparison - Generation {generation}<br><sub>Click on any point to download robot</sub>',
            xaxis_title='Branching',
            yaxis_title='Limbs',
            height=600,
            showlegend=True
        )

        return dcc.Graph(
            id='comparison-scatter',
            figure=fig
        )

    def run(self, host='127.0.0.1', port=8052, debug=True):
        """Run the dashboard server."""
        print(f"Starting Multiple Runs Dashboard at http://{host}:{port}")
        print(f"Analyzing {self.num_runs} independent runs")
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
            self.is_maximisation = resolved["is_maximisation"]
            self.first_generation_id = resolved["first_generation_id"]
            self.num_of_generations = resolved["num_of_generations"]
            self.target_population_size = resolved["target_population_size"]
            self.genotype = genotype
            self.task = resolved["task"]
            self.task_params = resolved["task_params"]
            self.genotype_name = genotype_name

    return DashboardConfig()


def load_populations_from_database(db_path: Path) -> Tuple[List[Population], Any]:
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
        raise FileNotFoundError(f"No configuration file found for database {db_path}")

    return populations, config


def main():
    """Main entry point for multiple runs dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Multiple runs analysis dashboard")
    parser.add_argument("--db_paths", nargs='+', default=["__data__/multiple_runs/run0.db",
                                                          "__data__/multiple_runs/run1.db",
                                                          "__data__/multiple_runs/run2.db",
                                                          "__data__/multiple_runs/run3.db",
                                                          "__data__/multiple_runs/run4.db"],
                       help="Paths to database files (one per run)")
    parser.add_argument("--names", nargs='+',
                       help="Names for runs (default: Run 1, Run 2, ...)")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8052, help="Dashboard port")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    # Load all runs
    runs_data = []
    for i, db_path in enumerate(args.db_paths):
        db_path = Path(db_path)
        populations, config = load_populations_from_database(db_path)
        runs_data.append((populations, config))
        console.log(f"Loaded run {i+1}: {len(populations)} generations")


    if not runs_data:
        raise ValueError("No runs loaded!")

    # Set run names
    if args.names and len(args.names) == len(runs_data):
        run_names = args.names
    else:
        run_names = [f"Run {i+1}" for i in range(len(runs_data))]

    console.log(f"Starting dashboard with {len(runs_data)} runs")
    dashboard = MultipleRunsDashboard(runs_data, run_names)
    dashboard.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == "__main__":
    main()
