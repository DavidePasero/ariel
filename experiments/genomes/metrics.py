import networkx as nx
import numpy as np
import zss

from ariel.utils.morphological_descriptor import MorphologicalMeasures


def compute_6d_descriptor(robot_graph: nx.DiGraph) -> np.ndarray:
    """Computes the 6D morphological descriptor vector."""
    measures = MorphologicalMeasures(robot_graph)
    try:
        P = measures.P if measures.is_2d else 0.0
    except AttributeError:
        P = 0.0

    return np.array([
        measures.B,  # Branching
        measures.L,  # Limbs
        measures.E,  # Extensiveness
        measures.S,  # Symmetry
        P,  # Proportion
        measures.J,  # Joints
    ])


def calculate_similarity_descriptor(
    graph_a: nx.DiGraph, graph_b: nx.DiGraph
) -> float:
    """
    Calculates distance based on the Euclidean distance between 6D descriptors.
    Lower value means higher similarity.
    """
    desc_a = compute_6d_descriptor(graph_a)
    desc_b = compute_6d_descriptor(graph_b)

    return np.linalg.norm(desc_a - desc_b)


# --- OPTIMIZED TREE EDIT DISTANCE LOGIC ---


def _nx_to_zss_node(graph: nx.DiGraph, node_idx) -> zss.Node:
    """
    Recursively converts a NetworkX node and its children to a zss.Node.
    """
    # 1. Get the label (module_type)
    # Adjust 'module_type' if your node attribute key is different (e.g., 'type')
    label = graph.nodes[node_idx].get("module_type", "None")

    # 2. Create the ZSS node
    z_node = zss.Node(label)

    # 3. Recursively add children
    # Note: We assume the DiGraph edges point Parent -> Child
    children = sorted(list(graph.successors(node_idx)))

    for child_idx in children:
        z_node.addkid(_nx_to_zss_node(graph, child_idx))

    return z_node


def calculate_similarity_ted(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> float:
    """
    Calculates distance using the Zhang-Shasha algorithm (via zss).
    Time Complexity: O(N^3) or better, vs NP-Hard for generic graph edit distance.
    """
    # 1. Find roots (Node with in_degree == 0)
    # This assumes the graph is a valid rooted tree (Single root)
    try:
        root_a = next(n for n, d in graph_a.in_degree() if d == 0)
        root_b = next(n for n, d in graph_b.in_degree() if d == 0)
    except StopIteration:
        # Fallback if graphs are empty or malformed
        return 0.0 if len(graph_a) == len(graph_b) == 0 else float("inf")

    # 2. Convert NetworkX graphs to ZSS trees
    tree_a = _nx_to_zss_node(graph_a, root_a)
    tree_b = _nx_to_zss_node(graph_b, root_b)

    # 3. Calculate Distance
    # simple_distance assumes:
    # - Insert cost: 1
    # - Delete cost: 1
    # - Update cost: 1 (if labels don't match), 0 (if they match)
    return zss.simple_distance(tree_a, tree_b)
