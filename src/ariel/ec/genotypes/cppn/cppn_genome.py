from __future__ import annotations
import random
import json
from ariel.ec.genotypes.genotype import Genotype
from ariel.ec.genotypes.cppn.node import Node
from ariel.ec.genotypes.cppn.connection import Connection
from ariel.ec.genotypes.cppn.activations import ACTIVATION_FUNCTIONS, FUNCTIONS_TO_NAMES, DEFAULT_ACTIVATION
import networkx as nx
from ariel.body_phenotypes.robogen_lite.decoders.cppn.cppn_best_first import MorphologyDecoderBestFirst
from ariel.ec.genotypes.cppn.id_manager import IdManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ariel.ec.mutations import CPPNMutator
    from ariel.ec.crossovers import CPPNCrossover
    
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
MAX_MODULES = 20
T, R = NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS
NUM_CPPN_INPUTS, NUM_CPPN_OUTPUTS = 6, 1 + T + R

class CPPN_genotype(Genotype):
    """A genome in the NEAT algorithm."""

    id_manager = IdManager(
        node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1, 
        innov_start=NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS - 1  # we have 78 conns (6 inputs * 13 outputs) --> last ID is 77 
    )

    def __init__(self,
                 nodes: dict[int, Node],
                 connections: dict[int, Connection],
                 fitness: float
                 ):

        self.nodes = nodes
        self.connections = connections
        self.fitness = fitness

    @staticmethod
    def _get_random_weight():
        return random.uniform(-1.0, 1.0)
    
    # Helper method for random biases
    @staticmethod
    def _get_random_bias():
        return random.uniform(-1.0, 1.0)

    @staticmethod
    def _get_random_activation():
        """Selects a random activation function from the available list."""
        return random.choice(list(ACTIVATION_FUNCTIONS.values()))
        
    def copy(self):
        """Returns a new CPPN_genotype object with identical, deep-copied gene sets."""
        
        # Deep copy nodes (Node class has a copy method)
        new_nodes = {
            _id: node.copy()
            for _id, node in self.nodes.items()
        }
        
        # Deep copy connections (Connection class has a copy method)
        new_connections = {
            innov_id: conn.copy() 
            for innov_id, conn in self.connections.items()
        }
        
        # Return a new CPPN_genotype instance
        return CPPN_genotype(new_nodes, new_connections, self.fitness)

    @classmethod
    def random(cls, 
               num_inputs: int, 
               num_outputs: int, 
               next_node_id: int, 
               next_innov_id: int):
        """
        Creates a new, randomly initialized CPPN_genotype with a base topology.
        Initial topology is fully connected inputs to outputs.
        """
        
        nodes = {}
        connections = {}
        
        # 1. Create Input Nodes
        for i in range(num_inputs):
            node = Node(_id=i, typ='input', activation=None, bias=0.0)
            nodes[i] = node
            
        # 2. Create Output Nodes (starting ID after inputs)
        current_node_id = num_inputs
        for _ in range(num_outputs):
            node = Node(
                _id=current_node_id, 
                typ='output', 
                activation=DEFAULT_ACTIVATION, 
                bias=cls._get_random_bias()
            )
            nodes[current_node_id] = node
            current_node_id += 1
            
        # 3. Create Connections (Fully connect inputs to outputs)
        current_innov_id = next_innov_id
        for in_id in range(num_inputs):
            for out_id in range(num_inputs, num_inputs + num_outputs):
                weight = cls._get_random_weight()
                connection = Connection(in_id, out_id, weight, enabled=True, innov_id=current_innov_id)
                connections[current_innov_id] = connection
                current_innov_id += 1 # Increment for the next unique innovation ID

        return cls(nodes, connections, fitness=0.0)

    @staticmethod
    def get_crossover_object() -> CPPNCrossover:
        """Return the crossover operator for this genotype type."""
        from ariel.ec.crossovers import CPPNCrossover
        return CPPNCrossover()
    
    @staticmethod
    def get_mutator_object() -> CPPNMutator:
        """Return the mutator operator for this genotype type."""
        from ariel.ec.mutations import CPPNMutator
        return CPPNMutator(CPPN_genotype.id_manager.get_next_innov_id, CPPN_genotype.id_manager.get_next_node_id)
    
    @staticmethod
    def create_individual(**kwargs: dict) -> "Genotype":
        """Generate a new individual of this genotype type."""

        num_inputs = NUM_CPPN_INPUTS
        num_outputs = NUM_CPPN_OUTPUTS
        next_node_id = CPPN_genotype.id_manager.get_next_node_id()
        next_innov_id = CPPN_genotype.id_manager.get_next_innov_id()
        
        return CPPN_genotype.random(num_inputs, num_outputs, next_node_id, next_innov_id)


    def add_connection(self, connection: Connection):
        in_id, out_id = connection.in_id, connection.out_id
        n_in, n_out = self.nodes[in_id], self.nodes[out_id]

        if n_out.typ == 'input':
            raise ValueError("Cannot connect into an input node.")
        if n_in.typ == 'output':
            raise ValueError("Cannot connect from an output node.")
        if in_id == out_id:
            raise ValueError("Self-loops are not allowed.")
        for c in self.connections.values():
            if c.enabled and c.in_id == in_id and c.out_id == out_id:
                raise ValueError("Connection already exists (same endpoints).")
        if self._would_create_cycle(in_id, out_id):
            raise ValueError(f"Adding {in_id}->{out_id} would create a cycle.")
        self.connections[connection.innov_id] = connection
    
    def add_node(self, node: Node):
        """Adds a node gene to the CPPN_genotype."""
        if node not in self.nodes.values():
            self.nodes[node._id] = node
        else:
            raise ValueError("Node already exists in CPPN_genotype.")
        
    def get_node_ordering(self):
        """
        Calculates a topological sort order for feed-forward activation using Kahn's algorithm.
        This ensures a node is evaluated only after all its input nodes are ready.
        https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
        """
        # 1. Build the graph structure and count incoming connections (in-degrees) for each node.
        graph = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections.values():
            if conn.enabled:
                # An edge goes from the input node to the output node
                graph[conn.in_id].append(conn.out_id)
                # The output node gains an incoming connection
                in_degree[conn.out_id] += 1

        # 2. Initialize a queue with all nodes that have no incoming connections.
        # These are the network's starting points (i.e., the input nodes).
        queue = [node_id for node_id in self.nodes if in_degree[node_id] == 0]
        
        sorted_order = []
        
        # 3. Process nodes in the queue.
        while queue:
            # Dequeue a node that is ready to be evaluated.
            node_id = queue.pop(0)
            sorted_order.append(node_id)

            # For the node we just processed, "remove" its outgoing edges.
            for neighbor_id in graph[node_id]:
                in_degree[neighbor_id] -= 1
                # If a neighbor's in-degree drops to 0, it's now ready to be evaluated.
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # 4. Final check: If the sorted order doesn't include all nodes,
        # this means there was a cycle in the graph (a recurrent connection).
        if len(sorted_order) != len(self.nodes):
            # For a feed-forward CPPN, this indicates an issue.
            raise Exception("A cycle was detected in the CPPN_genotype's graph, cannot create a feed-forward order.")
            
        return sorted_order
    
    def activate(self, inputs: list[float]) -> list[float]:
        """
        Activates the neural network by performing a forward pass with a given list of inputs.
        """
        
        node_outputs = {}
        ordered_node_ids = self.get_node_ordering()
        
        # 1. Initialize Input Node Outputs
        input_node_ids = [_id for _id, node in self.nodes.items() if node.typ == 'input']
        
        if len(inputs) != len(input_node_ids):
            raise ValueError(f"Expected {len(input_node_ids)} inputs, got {len(inputs)}")

        # Assign input values to input nodes based on order
        for i, node_id in enumerate(input_node_ids):
            node_outputs[node_id] = inputs[i]

        # 2. Activate Hidden and Output Nodes in order
        for node_id in ordered_node_ids:
            node = self.nodes[node_id]
            
            # Skip input nodes
            if node.typ == 'input':
                continue

            weighted_sum = 0.0
            
            # Find all connections where this node is the output
            for conn in self.connections.values():
                if conn.out_id == node_id and conn.enabled:
                    in_id = conn.in_id
                    # Ensure the input node has already been activated/initialized
                    if in_id in node_outputs:
                        weighted_sum += node_outputs[in_id] * conn.weight
            
            # Add bias
            weighted_sum += node.bias
    
            # Apply activation function
            node_outputs[node_id] = node.activation(weighted_sum)

        # 3. Collect Output Values from Output Nodes
        output_node_ids = [_id for _id, node in self.nodes.items() if node.typ == 'output']
        
        return [node_outputs[_id] for _id in output_node_ids]
    

    def to_digraph(self: "Genotype", **kwargs: dict) -> nx.DiGraph:
        """Convert the genotype to a directed graph representation."""
        decoder = MorphologyDecoderBestFirst(
            cppn_genome=self, 
            max_modules=MAX_MODULES
        )
        return decoder.decode()
    
    def _would_create_cycle(self, src_id: int, dst_id: int) -> bool:
        if src_id == dst_id:
            return True
        # Build adjacency of enabled edges
        adj = {n: [] for n in self.nodes}
        for c in self.connections.values():
            if c.enabled:
                adj[c.in_id].append(c.out_id)
        # If dst can reach src already, adding src->dst closes a cycle
        stack = [dst_id]
        seen = set()
        while stack:
            u = stack.pop()
            if u == src_id:
                return True
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj.get(u, []))
        return False
    


    # --- JSON Serialization Methods --- #
    def to_json(self, **kwargs) -> str:
        """Serialize genotype to a JSON string."""

        data = {
            "fitness": self.fitness,
            "nodes": [
                {
                    "id": node_id,
                    "type": node.typ,
                    # input nodes typically have no activation; others do
                    "activation":   None
                                    if self.nodes[node_id].activation is None
                                    else FUNCTIONS_TO_NAMES.get(node.activation, getattr(node.activation, "__name__", None)),
                    "bias": self.nodes[node_id].bias,
                }
                for node_id, node in self.nodes.items()
            ],
            "connections": [
                {
                    "innov_id": innov_id,
                    "in": conn.in_id,
                    "out": conn.out_id,
                    "weight": conn.weight,
                    "enabled": conn.enabled,
                }
                for innov_id, conn in self.connections.items()
            ],
        }
        return json.dumps(data)

    @staticmethod
    def from_json(json_data: str, **kwargs) -> "Genotype":
        """Deserialize genotype from a JSON string produced by to_json()."""

        obj = json.loads(json_data)

        # Rebuild nodes
        nodes: dict[int, Node] = {}
        for n in obj["nodes"]:
            n_id = int(n["id"])
            typ = n["type"]
            if typ == "input":
                act = None
            else:
                act_name = n.get("activation")
                act = ACTIVATION_FUNCTIONS.get(act_name, DEFAULT_ACTIVATION)
            bias = float(n.get("bias", 0.0))
            nodes[n_id] = Node(_id=n_id, typ=typ, activation=act, bias=bias)

        # Rebuild connections
        connections: dict[int, Connection] = {}
        for c in obj["connections"]:
            innov_id = int(c["innov_id"])
            connections[innov_id] = Connection(
                int(c["in"]),
                int(c["out"]),
                float(c["weight"]),
                bool(c.get("enabled", True)),
                innov_id=innov_id,
            )

        fitness = float(obj.get("fitness", 0.0))
        return CPPN_genotype(nodes, connections, fitness)