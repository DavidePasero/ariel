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
from pathlib import Path
from typing import List, Callable, Any
from networkx.readwrite import json_graph

from plotly_morphology_analysis import PlotlyMorphologyAnalyzer
from ariel.ec.a001 import Individual

type Population = List[Individual]


class EvolutionDashboard:
    """Interactive dashboard for evolution visualization."""

    def __init__(self, populations: List[Population], decoder: Callable, config: Any):
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
        self.analyzer.load_target_robots(str(config.task_params["target_robot_path"]))

        # Pre-compute fitness timeline
        self._compute_fitness_timeline()

        # Cache for computed descriptors per generation
        self._descriptor_cache = {}

        # Cache for individual data per generation (for click handling)
        self._individual_cache = {}

        # Morphological feature names
        self.feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        # Create dl_robots directory if it doesn't exist
        self.dl_robots_path = Path("examples/dl_robots")
        self.dl_robots_path.mkdir(exist_ok=True)

        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()

    def _compute_fitness_timeline(self):
        """Compute average fitness for each generation."""
        self.fitness_timeline = []

        for gen_idx, population in enumerate(self.populations):
            if population:
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                self.fitness_timeline.append({
                    'generation': gen_idx,
                    'avg_fitness': avg_fitness,
                    'best_fitness': max(ind.fitness for ind in population),
                    'worst_fitness': min(ind.fitness for ind in population)
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
            'descriptors': self.analyzer.descriptors.copy(),
            'fitness_scores': self.analyzer.fitness_scores.copy()
        }

        # Cache individual data for click handling
        self._individual_cache[generation] = population.copy()

        return self._descriptor_cache[generation]

    def _setup_layout(self):
        """Setup the dashboard layout."""
        max_generation = len(self.populations) - 1

        self.app.layout = html.Div([
            html.H1("Evolution Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),

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
                    max=max_generation,
                    step=1,
                    value=max_generation,
                    marks={i: str(i) for i in range(0, max_generation + 1, max(1, max_generation // 10))},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Fitness timeline (always visible)
            html.Div([
                html.H3("Fitness Over Generations"),
                dcc.Graph(id='fitness-timeline')
            ], style={'margin': '20px', 'marginBottom': 40}),

            # Tabbed plots section
            html.Div([
                dcc.Tabs(id='plot-tabs', value='single-feature-tab', children=[
                    dcc.Tab(label='Single Feature', value='single-feature-tab'),
                    dcc.Tab(label='Fitness Landscapes', value='landscape-tab'),
                    dcc.Tab(label='Fitness Distributions', value='distribution-tab'),
                    dcc.Tab(label='Morphological Diversity', value='diversity-tab'),
                    dcc.Tab(label='Pairwise Features', value='pairwise-tab'),
                ]),
                html.Div(id='tab-content')
            ], style={'margin': '20px'})
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            Output('fitness-timeline', 'figure'),
            Input('generation-slider', 'value')
        )
        def update_fitness_timeline(selected_generation):
            """Update fitness timeline with highlighted generation."""
            if not self.fitness_timeline:
                return go.Figure()

            df = pd.DataFrame(self.fitness_timeline)

            fig = go.Figure()

            # Add average fitness line
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['avg_fitness'],
                mode='lines+markers',
                name='Average Fitness',
                line=dict(color='blue', width=2)
            ))

            # Add best fitness line
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['best_fitness'],
                mode='lines+markers',
                name='Best Fitness',
                line=dict(color='green', width=2)
            ))

            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(go.Scatter(
                    x=[selected_row['generation']],
                    y=[selected_row['avg_fitness']],
                    mode='markers',
                    name=f'Generation {selected_generation}',
                    marker=dict(color='red', size=12, symbol='circle-open', line=dict(width=3))
                ))

            fig.update_layout(
                title='Fitness Evolution Over Generations',
                xaxis_title='Generation',
                yaxis_title='Fitness',
                height=400,
                showlegend=True
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
            return self._plot_feature_evolution(feature_idx)

        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('plot-tabs', 'value'),
             Input('generation-slider', 'value')]
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            # Get data for selected generation
            gen_data = self._get_generation_data(selected_generation)

            if active_tab == 'landscape-tab':
                return self._create_landscape_plot(selected_generation)
            elif active_tab == 'distribution-tab':
                return self._create_distribution_plot(selected_generation)
            elif active_tab == 'diversity-tab':
                return self._create_diversity_plot(selected_generation)
            elif active_tab == 'pairwise-tab':
                return self._create_pairwise_plot(selected_generation)
            elif active_tab == 'single-feature-tab':
                return self._create_single_feature_plot()

            return html.Div("Select a tab to view plots")

        # Add click callbacks for different graph types
        @self.app.callback(
            Output('status-message', 'children'),
            Output('status-message', 'style'),
            [Input('landscape-graph', 'clickData'),
             Input('pairwise-graph', 'clickData')],
            [State('generation-slider', 'value')],
            prevent_initial_call=True
        )
        def handle_graph_clicks(landscape_click, pairwise_click, generation):
            """Handle clicks on any graph."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return "", {'display': 'none'}

            # Determine which input triggered the callback
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'landscape-graph' and landscape_click:
                return self._handle_point_click(landscape_click, generation)
            elif trigger_id == 'pairwise-graph' and pairwise_click:
                return self._handle_point_click(pairwise_click, generation)

            return "", {'display': 'none'}

    def _create_landscape_plot(self, generation: int):
        """Create fitness landscape plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_fitness_landscapes()
        fig.update_layout(title_text=f"Fitness Landscapes - Generation {generation}")

        # Create graph with click handling
        return dcc.Graph(
            id='landscape-graph',
            figure=fig
        )

    def _handle_point_click(self, clickData, generation):
        """Handle click on a data point to save robot."""
        if not clickData or 'points' not in clickData:
            return "", {'display': 'none'}

        try:
            # Get the clicked point index
            point = clickData['points'][0]
            point_index = point.get('pointIndex', point.get('pointNumber', 0))

            # Get the individual from cache
            if generation not in self._individual_cache:
                return "Error: Generation data not found", {
                    'display': 'block',
                    'backgroundColor': '#ffcccc',
                    'color': 'red',
                    'textAlign': 'center',
                    'marginBottom': 20,
                    'padding': '10px',
                    'borderRadius': '5px'
                }

            population = self._individual_cache[generation]
            if point_index >= len(population):
                return f"Error: Point index {point_index} out of range", {
                    'display': 'block',
                    'backgroundColor': '#ffcccc',
                    'color': 'red',
                    'textAlign': 'center',
                    'marginBottom': 20,
                    'padding': '10px',
                    'borderRadius': '5px'
                }

            # Get the individual and convert to robot graph
            individual = population[point_index]
            robot_graph = self.decoder(individual)

            # Save robot as JSON
            robot_data = json_graph.node_link_data(robot_graph, edges="edges")

            # Save to file
            filename = f"robot_gen{generation}_ind{point_index}_fit{individual.fitness:.3f}.json"
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

    def _create_distribution_plot(self, generation: int):
        """Create fitness distribution plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_fitness_distributions()
        fig.update_layout(title_text=f"Fitness Distributions - Generation {generation}")
        return dcc.Graph(figure=fig)

    def _create_diversity_plot(self, generation: int):
        """Create morphological diversity analysis plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.analyze_morphological_diversity()
        fig.update_layout(title_text=f"Morphological Diversity - Generation {generation}")
        return dcc.Graph(figure=fig)

    def _create_pairwise_plot(self, generation: int):
        """Create pairwise feature landscape plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_pairwise_feature_landscapes()
        fig.update_layout(title_text=f"Pairwise Feature Landscapes - Generation {generation}")

        # Create graph with click handling
        return dcc.Graph(
            id='pairwise-graph',
            figure=fig
        )

    def _create_single_feature_plot(self):
        """Create single feature evolution plot with feature selector."""
        return html.Div([
            html.Div([
                html.Label("Select Feature:", style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': feature, 'value': i} for i, feature in enumerate(self.feature_names)],
                    value=0,
                    style={'marginBottom': 20}
                )
            ], style={'width': '300px', 'margin': '0 auto'}),
            dcc.Graph(id='single-feature-graph')
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
            if gen_data['descriptors'].size > 0:
                # Extract the specific feature for this generation
                feature_values = gen_data['descriptors'][:, feature_idx]

                # Find best individual (closest to any target)
                best_value = feature_values[0]
                if hasattr(self.analyzer, 'target_descriptors') and len(self.analyzer.target_descriptors) > 0:
                    # Find individual closest to any target for this feature
                    min_distance = float('inf')
                    for target_desc in self.analyzer.target_descriptors:
                        target_value = target_desc[feature_idx]
                        distances = np.abs(feature_values - target_value)
                        closest_idx = np.argmin(distances)
                        if distances[closest_idx] < min_distance:
                            min_distance = distances[closest_idx]
                            best_value = feature_values[closest_idx]

                generation_data.append({
                    'generation': gen_idx,
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'median': np.median(feature_values),
                    'best': best_value
                })

        if not generation_data:
            return go.Figure()

        df = pd.DataFrame(generation_data)

        fig = go.Figure()

        # Add mean line with error bands
        fig.add_trace(go.Scatter(
            x=df['generation'],
            y=df['mean'],
            mode='lines+markers',
            name='Population Mean',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Add standard deviation bands
        fig.add_trace(go.Scatter(
            x=df['generation'].tolist() + df['generation'][::-1].tolist(),
            y=(df['mean'] + df['std']).tolist() + (df['mean'] - df['std'])[::-1].tolist(),
            fill='tonext',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='±1 Standard Deviation'
        ))

        # Add median line
        fig.add_trace(go.Scatter(
            x=df['generation'],
            y=df['median'],
            mode='lines+markers',
            name='Population Median',
            line=dict(color='purple', width=2, dash='dot'),
            marker=dict(size=4)
        ))

        # Add population best line (closest to target)
        fig.add_trace(go.Scatter(
            x=df['generation'],
            y=df['best'],
            mode='lines+markers',
            name='Population Best (Closest to Target)',
            line=dict(color='green', width=2),
            marker=dict(size=5, symbol='star')
        ))

        # Add target robot values as horizontal lines
        if hasattr(self.analyzer, 'target_descriptors') and len(self.analyzer.target_descriptors) > 0:
            for i, (target_desc, target_name) in enumerate(zip(self.analyzer.target_descriptors, self.analyzer.target_names)):
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
            title=f'{feature_name} Evolution Over Generations',
            xaxis_title='Generation',
            yaxis_title=f'{feature_name} Value',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )

        return fig

    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard server."""
        print(f"Starting Evolution Dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def run_dashboard(populations: List[Population], decoder: Callable, config: Any,
                 host='127.0.0.1', port=8050, debug=True):
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


if __name__ == "__main__":
    # Example usage - this would be called from evolve.py
    print("Evolution Dashboard module - import and call run_dashboard() with your evolution data")
