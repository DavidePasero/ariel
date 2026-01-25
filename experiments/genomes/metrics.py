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
        # measures.num_bricks,
        # measures.num_active_hinges,
    ])


def calculate_similarity_descriptor(
    individual: nx.DiGraph, target: nx.DiGraph
) -> float:
    """
    Calculates distance based on the Euclidean distance between 6D descriptors.
    Lower value means higher similarity.
    """
    desc_ind = compute_6d_descriptor(individual)
    desc_target = compute_6d_descriptor(target)
    # Normalize by number of bricks and hinges to avoid bias towards larger robots
    if len(desc_ind) == 8:
        desc_ind[6] = desc_ind[6] / desc_target[6]
        desc_target[6] = 1.0
        desc_ind[7] = desc_ind[7] / desc_target[7]
        desc_target[7] = 1.0

    return np.linalg.norm(desc_ind - desc_target)


# --- OPTIMIZED TREE EDIT DISTANCE LOGIC ---


def _nx_to_zss_node(graph: nx.DiGraph, node_idx) -> zss.Node:
    """
    Recursively converts a NetworkX node and its children to a zss.Node.
    """
    label = graph.nodes[node_idx].get("module_type", "None")
    z_node = zss.Node(label)

    # We sort children to ensure deterministic order if the graph structure is ambiguous
    children = sorted(list(graph.successors(node_idx)))
    for child_idx in children:
        z_node.addkid(_nx_to_zss_node(graph, child_idx))
    return z_node


def calculate_similarity_ted(
    individual: nx.DiGraph, target_graph: nx.DiGraph
) -> float:
    """
    Calculates Tree Edit Distance normalized by the size of the target graph.
    """
    # 0. Handle empty graphs
    len_a = len(individual)
    len_b = len(target_graph)

    if len_b == 0:
        if len_a == 0:
            return 0.0  # Both empty -> Identical
        else:
            return float(
                "inf"
            )  # Infinite distance if target is empty but source is not

    # 1. Find roots
    try:
        root_a = next(n for n, d in individual.in_degree() if d == 0)
        root_b = next(n for n, d in target_graph.in_degree() if d == 0)
    except StopIteration:
        # Handle cases where graphs might be cyclic or have no clear root
        return float("inf")

    # 2. Convert NetworkX graphs to ZSS trees
    tree_a = _nx_to_zss_node(individual, root_a)
    tree_b = _nx_to_zss_node(target_graph, root_b)

    # Print lables for debugging
    # print("Tree A labels:", tree_a.label)

    # 3. Calculate Raw Distance
    raw_distance = zss.simple_distance(tree_a, tree_b)

    # 4. Normalize
    # This tells you: "How much error is there per node in the target?"
    normalized_score = raw_distance / len_b

    return normalized_score
