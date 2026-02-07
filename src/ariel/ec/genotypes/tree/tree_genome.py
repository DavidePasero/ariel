from __future__ import annotations
import json
#import ariel.body_phenotypes.robogen_lite.config as config
from collections import deque
import networkx as nx
from collections.abc import Callable
from functools import reduce
import numpy as np
from typing import TYPE_CHECKING
import uuid
import random
from networkx.readwrite import json_graph

from ariel.ec.genotypes.genotype import Genotype

if TYPE_CHECKING:
    from ariel.ec.crossovers import TreeCrossover
    from ariel.ec.mutations import TreeMutator

SEED = 42
RNG = np.random.default_rng(SEED)



# ==========================================
# 2. Tree Genome Class
# ==========================================

class TreeGenome(Genotype):
    VALID_TYPES = {"C", "B", "H"}
    VALID_ROTATIONS = {0, 45, 90, 135, 180, 225, 270, 315}
    VALID_FACES = {"FRONT", "BACK", "TOP", "BOTTOM", "LEFT", "RIGHT"}

    def __init__(self, root: TreeNode | None = None):
        self.tree = nx.DiGraph()
        self.root_id=str(uuid.uuid4())
        self.tree.add_node(self.root_id, type="C", rotation=0)

    @staticmethod
    def get_crossover_object() -> TreeCrossover:
        from ariel.ec.crossovers import TreeCrossover

        """Return the crossover operator for tree genomes."""
        return TreeCrossover()

    @staticmethod
    def get_mutator_object() -> TreeMutator:
        from ariel.ec.mutations import TreeMutator

        """Return the mutator operator for tree genomes."""
        return TreeMutator()

    @staticmethod
    def create_individual(**kwargs: dict) -> TreeGenome:
        """Generate a new TreeGenome individual."""
        return TreeGenome.default_init()

    @classmethod
    def default_init(cls, *args, **kwargs):
        """Default instantiation with a core root."""
        return cls()

    def is_face_occupied(self, parent_id, face):
        """
        Checks if a specific face on the parent node already has a connection.
        """
        if parent_id not in self.tree:
            raise KeyError(f"Node {parent_id} does not exist.")
        
        # Look at all outgoing edges from the parent
        for _, _, edge_data in self.tree.out_edges(parent_id, data=True):
            if edge_data.get("face") == face:
                return True
        return False

    def add_node(self, node_type, rotation, parent_id, face):
        """
        Adds a node only if the target face on the parent is available.
        """
        if self.is_face_occupied(parent_id, face):
            raise ValueError(f"Conflict: Face '{face}' on node '{parent_id}' is already occupied.")

        if node_type not in self.VALID_TYPES:
            raise ValueError(f"Type must be {self.VALID_TYPES}")
        
        if rotation not in self.VALID_ROTATIONS:
            raise ValueError(f"Rotation must be {self.VALID_ROTATIONS}")
        node_id = str(uuid.uuid4())
        self.tree.add_node(node_id, type=node_type, rotation=rotation)
        self.tree.add_edge(parent_id, node_id, face=face)
        return node_id

    def remove_node(self, node_id):
        if node_id == self.root:
            raise ValueError("Cannot remove root.")
        if node_id in self.tree:
            descendants = list(nx.descendants(self.tree, node_id))
            self.tree.remove_nodes_from(descendants + [node_id])

    def extract_subtree(self, node_id):
        """
        Creates a new independent StructuralGenome starting from node_id.
        Useful for crossover operations.
        """
        if node_id not in self.tree:
            raise KeyError(f"Node {node_id} not found.")

        # 1. Identify all nodes in the subtree
        subtree_nodes = list(nx.descendants(self.tree, node_id)) + [node_id]
        
        # 2. Create a subgraph and deep copy it
        subgraph = self.tree.subgraph(subtree_nodes).copy()
        
        id_mapping = {old: str(uuid.uuid4()) for old in subtree_nodes}
        new_root_id = id_mapping[node_id]
        
        # 4. Relabel the nodes in the subgraph
        new_graph = nx.relabel_nodes(subgraph, id_mapping, copy=True)

        # 3. Initialize a new StructuralGenome instance
        # We fetch the root node's data for the new tree
        new_genome = TreeGenome()
        
        # 4. Overwrite the new_genome's tree with our extracted subgraph
        new_genome.tree = new_graph
        new_genome.root_id = new_root_id
        
        return new_genome

    def remove_subtree(self, node_id):
        """
        Removes the specified node and all its descendants.
        This effectively 'prunes' the branch from the genome.
        """
        if node_id not in self.tree:
            raise KeyError(f"Node '{node_id}' not found in the tree.")

        if node_id == self.root_id:
            raise ValueError(
                "Cannot remove the root node via remove_subtree. "
                "If you want to clear the tree, re-initialize the object."
            )

        # 1. Find all descendants (children, grandchildren, etc.)
        # nx.descendants returns a set of all nodes reachable from node_id
        nodes_to_remove = list(nx.descendants(self.tree, node_id))
        
        # 2. Add the node itself to the list
        nodes_to_remove.append(node_id)

        # 3. Remove all identified nodes from the graph
        # This also automatically removes any edges connected to these nodes
        self.tree.remove_nodes_from(nodes_to_remove)

    def get_node_on_face(self, parent_id, face):
        """
        Returns the ID of the node connected to the specified face of parent_id.
        Returns None if the face is unoccupied.
        """
        if parent_id not in self.tree:
            raise KeyError(f"Node {parent_id} not found in the genome.")

        if face not in self.VALID_FACES:
            raise ValueError(f"Invalid face name. Must be one of {self.VALID_FACES}")

        # Iterate through outgoing edges from the parent
        for _, child_id, edge_data in self.tree.out_edges(parent_id, data=True):
            if edge_data.get("face") == face:
                return child_id
        
        return None


    def graft_subtree(self, subtree_genome, parent_id, face):
        """
        Grafts an existing subtree into this genome.
        Assumes node IDs are unique UUIDs.
        """
        # 1. Basic Safety Checks
        if parent_id not in self.tree:
            raise KeyError(f"Parent node '{parent_id}' not found.")
        
        if self.is_face_occupied(parent_id, face):
            raise ValueError(f"Face '{face}' on node '{parent_id}' is already occupied.")
        # 2. Prepare the subtree root
        # Update the rotation of the subtree's root to match the new connection point
        subtree_root_id = subtree_genome.root_id

        # 3. Merge the subtree graph into the current tree
        # Since IDs are UUIDs, this will add new nodes/edges without overwriting
        self.tree = nx.compose(self.tree, subtree_genome.tree)

        # 4. Create the structural link (the 'Face' connection)
        self.tree.add_edge(parent_id, subtree_root_id, face=face)

    def graft_subtree_inplace(self, subtree_genome, target_node_id):
        """
        Removes the target_node_id and its entire subtree, then grafts 
        the new subtree_genome onto the same parent and face.
        """
        if target_node_id == self.root_id:
            raise ValueError("Cannot replace the root node in-place. Root must remain 'C'.")

        if target_node_id not in self.tree:
            raise KeyError(f"Node {target_node_id} not found.")

        # 1. Identify the parent and the connection face before we delete the node
        # In a tree, every node (except root) has exactly one predecessor
        parent_id = list(self.tree.predecessors(target_node_id))[0]
        edge_data = self.tree.get_edge_data(parent_id, target_node_id)
        original_face = edge_data['face']

        # 2. Remove the existing subtree at that location
        # We use the previous remove_node logic to clean up descendants
        self.remove_subtree(target_node_id)

        # 3. Use our existing graft logic to plug the new genome into the hole
        # We use the original_face so the new limb 'points' the same direction
        self.graft_subtree(subtree_genome, parent_id, original_face)
        

    @staticmethod
    def create_random_subtree(max_depth=3, branching_prob=0.5,existing_nodes=0,max_modules=32):
        """
        Creates a new StructuralGenome composed of B and H nodes.
        branching_prob: 0.0 to 1.0 chance that each available face will grow a new node.
        """
        # 1. Randomly pick root type (B or H)
        root_type = random.choice(["B", "H"])
        root_id = str(uuid.uuid4())
        root_rotation = random.choice(list(TreeGenome.VALID_ROTATIONS))
        
        new_genome = TreeGenome()
        new_genome.tree = nx.DiGraph()
        new_genome.root_id=root_id
        new_genome.tree.add_node(root_id, type=root_type, rotation=root_rotation)

        
        # 2. Start recursive growth
        new_genome._grow_random_recursive(root_id, 1, max_depth, branching_prob,existing_nodes,max_modules)
        
        return new_genome
    @staticmethod
    def create_random_tree(max_depth=3, branching_prob=0.5,existing_nodes=0,max_modules=32):
        """
        Creates a new StructuralGenome composed of B and H nodes.
        branching_prob: 0.0 to 1.0 chance that each available face will grow a new node.
        """
        # 1. Randomly pick root type (B or H)
        root_type = "C"
        root_id = str(uuid.uuid4())
        root_rotation = 0
        
        new_genome = TreeGenome()
        new_genome.tree = nx.DiGraph()
        new_genome.root_id=root_id
        new_genome.tree.add_node(root_id, type=root_type, rotation=root_rotation)

        
        # 2. Start recursive growth
        new_genome._grow_random_recursive(root_id, 1, max_depth, branching_prob,existing_nodes,max_modules)
        
        return new_genome

    def get_random_node(self):
        """
        Returns the ID of a random node in the tree, excluding the root.
        Returns None if the tree only contains the root.
        """
        # Get all node IDs from the NetworkX graph
        all_nodes = list(self.tree.nodes)
        
        # Filter out the root ID
        non_root_nodes = [node for node in all_nodes if node != self.root_id]
        
        if not non_root_nodes:
            print("Warning: Tree only contains the root node.")
            return None
            
        return random.choice(non_root_nodes)    

    def _grow_random_recursive(self, parent_id, depth, max_depth, branching_prob,existing_modules,max_modules):
        if depth >= max_depth:
            return

        # Iterate through every possible face to decide if it sprouts a new component
        available_faces = list(self.VALID_FACES)
        random.shuffle(available_faces)

        for face in available_faces:
            # Check branching probability
            if random.random() < branching_prob:
                # Check if face is already occupied (just in case)
                if not self.is_face_occupied(parent_id, face):
                    node_type = random.choice(["B", "H"])
                    rotation = random.choice(list(self.VALID_ROTATIONS))
                    
                    # Add node and recurse
                    if existing_modules+1<max_modules:
                        new_node_id = self.add_node(node_type, rotation, parent_id, face)
                        self._grow_random_recursive(new_node_id, depth + 1, max_depth, branching_prob,existing_modules+1,max_modules)

    def display_tree(self, current_node=None, indent="", is_last=True, prefix="Root"):
        """
        Recursively prints the tree structure in a human-readable format.
        """
        if current_node is None:
            current_node = self.root_id

        # Retrieve node data
        node_data = self.tree.nodes[current_node]
        node_type = node_data.get('type', '?')
        rotation = node_data.get('rotation', 0)

        # Create the visual marker for the current branch
        marker = "└── " if is_last else "├── "
        
        # Print the current node's info
        print(f"{indent}{marker}[{prefix}] Node: {current_node[:8]}... Type: {node_type}, Rot: {rotation}°")

        # Prepare indentation for children
        child_indent = indent + ("    " if is_last else "│   ")
        
        # Get all outgoing edges and their 'face' attribute
        children = list(self.tree.out_edges(current_node, data=True))
        
        for i, (parent, child, edge_data) in enumerate(children):
            face = edge_data.get('face', 'UNKNOWN')
            is_last_child = (i == len(children) - 1)
            # Recurse into the child node
            self.display_tree(child, child_indent, is_last_child, prefix=face)
    
    def to_digraph(self):
        """Returns the underlying NetworkX DiGraph object."""
        # Returns a copy to prevent accidental external modification
        return self.tree.copy()

    @staticmethod
    def from_json(json_str: str, **kwargs) -> "TreeGenome":
        """Creates a new StructuralGenome instance from a JSON string."""
        data = json.loads(json_str)
        
        # Reconstruct the NetworkX graph object
        graph = json_graph.node_link_graph(data["graph_data"])
        
        # Create instance and restore state
        root_id = data["root_id"]
        # We initialize with placeholder data and then overwrite with the real tree
        instance = TreeGenome()
        instance.root_id=root_id
        instance.tree = graph
        
        return instance

    def to_json(self, *, indent: int | None = 2) -> str:
        """Converts the genome into a JSON-formatted string."""
        # Convert the NetworkX structure to a node-link dictionary
        graph_dict = json_graph.node_link_data(self.tree)
        
        # Package with the root_id metadata
        full_package = {
            "root_id": self.root_id,
            "graph_data": graph_dict
        }
        # Return as a serialized string
        return json.dumps(full_package)

    def copy(self):
        """
        Creates a completely independent clone of the current genome.
        Any changes made to the clone will NOT affect the original.
        """
        # 1. Create a new instance with the same root ID
        # Note: Since the tree structure is copied, we just need to initialize the object.
        new_genome = TreeGenome()
        
        # 2. Use NetworkX's built-in deep copy for the graph
        # This copies all nodes, edges, and their associated attribute dictionaries.
        new_genome.tree = self.tree.copy()
        
        # 3. Ensure the root pointer is identical
        new_genome.root_id = self.root_id
        
        return new_genome

    def get_subtree_size(self, node_id):
        """
        Calculates the total number of nodes in the subtree rooted at node_id.
        This includes the node itself and all nodes branching off from it.
        """
        if node_id not in self.tree:
            raise KeyError(f"Node {node_id} not found in the genome.")

        # nx.descendants returns a set of all nodes reachable from node_id
        # We add 1 to include the node_id itself
        return len(nx.descendants(self.tree, node_id)) + 1

def main():
    print("--- 1. Creating Initial Genome (Parent A) ---")
    parent_a = TreeGenome()
    # Build a small branch: C -> B -> H
    b_node = parent_a.add_node("B", 90, parent_a.root_id, "TOP")
    parent_a.add_node("H", 0, b_node, "FRONT")
    parent_a.display_tree()

    print("\n--- 2. Extracting Subtree (The B-H branch) ---")
    genetic_branch = parent_a.extract_subtree(b_node)
    genetic_branch.display_tree()
    parent_a.display_tree()

    print("\n--- 3. Creating Second Genome (Parent B) ---")
    parent_b = TreeGenome()
    # Add a base node to Parent B
    base_b = parent_b.add_node("B", 180, parent_b.root_id, "BOTTOM")
    parent_b.display_tree()


    print("\n--- 4. Grafting Branch from A onto Parent B ---")
    # Graft the extracted branch onto the new base on a different face
    parent_b.graft_subtree_inplace(genetic_branch, base_b)
    parent_b.display_tree()

    print("\n--- 5. Create a random subtree ---")
    random_subtree = TreeGenome.create_random_subtree()
    random_subtree.display_tree()

    print("\n--- 6. Mutate Parent B ---")
    mut=TreeGenome.get_mutator_object()
    mut.which_mutation="random_subtree_replacement"
    mut(parent_b)
    parent_b.display_tree()

    print("\n--- 7. Clone Mutated Parent B ---")
    clone_b=parent_b.copy()
    clone_b.display_tree()
    
    print("\n--- 8. Crossover Parent A and Parent B ---")
    parent_a.display_tree()
    parent_b.display_tree()
    cross=TreeGenome.get_crossover_object()
    cross.which_crossover="normal"
    child1,child2=cross(parent_a,parent_b)
    child1.display_tree()
    child2.display_tree()
    

if __name__ == "__main__":
    main()