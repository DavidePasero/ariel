"""TODO(jmdm): description of script."""

from __future__ import annotations  # noqa: I001

# Standard library
from abc import ABC
from pathlib import Path
from typing import cast, TYPE_CHECKING
import random
from functools import partial

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install
import copy
from ariel.ec.genotypes.cppn.node import Node
from ariel.ec.genotypes.cppn.connection import Connection


if TYPE_CHECKING:
    from ariel.ec.genotypes.genotype import Genotype
import ariel.body_phenotypes.robogen_lite.config as pheno_config

from ariel.ec.genotypes.genotype import MAX_MODULES
# Max attempts to find valid mutation points
MAX_ATTEMPTS = 20

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)
DB_NAME = "database.db"
DB_PATH: Path = DATA / DB_NAME
SEED = 42

# Global functions
install(width=180)
console = Console()
RNG = np.random.default_rng(SEED)


class Mutation(ABC):
    mutations_mapping: dict[str, function] = NotImplemented
    which_mutation: str = ""

    @classmethod
    def set_which_mutation(cls, mutation_type: str) -> None:
        cls.which_mutation = mutation_type

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.mutations_mapping = {
            name: getattr(cls, name)
            for name, val in cls.__dict__.items()
            if isinstance(val, staticmethod)
        }

    def __call__(
        self,
        individual: Genotype,
        **kwargs: dict,
    ) -> Genotype:
        """Perform crossover on two genotypes.

        Parameters
        ----------
        parent_i : Genotype
            The first parent genotype (list or nested list of integers).
        parent_j : Genotype
            The second parent genotype (list or nested list of integers).

        Returns
        -------
        tuple[Genotype, Genotype]
            Two child genotypes resulting from the crossover.
        """
        if self.which_mutation not in self.mutations_mapping:
            msg = f"Mutation type '{self.which_mutation}' not recognized."
            raise ValueError(msg)
        return self.mutations_mapping[self.which_mutation](individual, **kwargs)


class IntegerMutator(Mutation):
    @staticmethod
    def random_swap(
        individual: Genotype,
        low: int,
        high: int,
        mutation_probability: float,
    ) -> Genotype:
        shape = np.asarray(individual).shape
        mutator = RNG.integers(
            low=low,
            high=high,
            size=shape,
            endpoint=True,
        )
        mask = RNG.choice(
            [True, False],
            size=shape,
            p=[mutation_probability, 1 - mutation_probability],
        )
        new_genotype = np.where(mask, mutator, individual).astype(int).tolist()
        return cast("Integers", new_genotype.astype(int).tolist())

    @staticmethod
    def integer_creep(
        individual: Genotype,
        span: int,
        mutation_probability: float,
    ) -> Genotype:
        # Prep
        ind_arr = np.array(individual)
        shape = ind_arr.shape

        # Generate mutation values
        mutator = RNG.integers(
            low=1,
            high=span,
            size=shape,
            endpoint=True,
        )

        # Include negative mutations
        sub_mask = RNG.choice(
            [-1, 1],
            size=shape,
        )

        # Determine which positions to mutate
        do_mask = RNG.choice(
            [1, 0],
            size=shape,
            p=[mutation_probability, 1 - mutation_probability],
        )
        mutation_mask = mutator * sub_mask * do_mask
        new_genotype = ind_arr + mutation_mask
        return cast("Integers", new_genotype.astype(int).tolist())


class TreeMutator(Mutation):
    @staticmethod
    def _random_tree(
        max_depth: int = 2, branching_prob: float = 0.5
    ) -> Genotype:
        """Generate a random tree with pheno_configurable branching probability."""
        from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode

        genome = TreeGenome.default_init()  # Start with CORE
        face = RNG.choice(genome.root.available_faces())
        subtree = TreeNode.random_tree_node(
            max_depth=max_depth - 1, branch_prob=branching_prob
        )
        if subtree:
            genome.root._set_face(face, subtree)
        return genome

    @staticmethod
    def random_subtree_replacement(
        individual: Genotype,
        max_subtree_depth: int = 2,
        branching_prob: float = 0.5,
    ) -> Genotype:
        """Replace a random subtree with a new random subtree."""
        from ariel.ec.genotypes.tree.tree_genome import TreeNode

        new_individual = copy.copy(individual)

        # Collect all nodes in the tree
        all_nodes = new_individual.root.get_all_nodes(exclude_root=True)

        if not all_nodes:
            # print("Tree has no nodes to replace; generating a new random tree.")
            return TreeMutator._random_tree(
                max_depth=max_subtree_depth, branching_prob=branching_prob
            )

        # Generate a new random subtree
        new_subtree = TreeNode.random_tree_node(
            max_depth=max_subtree_depth, branch_prob=branching_prob
        )

        i=0
        node_to_replace = RNG.choice(all_nodes)
        while node_to_replace.num_descendants + 1 + new_individual.num_modules > MAX_MODULES:
            node_to_replace = RNG.choice(all_nodes)
            i += 1
            if i >= MAX_ATTEMPTS: return new_individual

        with new_individual.root.enable_replacement():
            new_individual.root.replace_node(node_to_replace, new_subtree)

        return new_individual


class LSystemMutator(Mutation):
    @staticmethod
    def mutate_one_point_lsystem(lsystem, mutation_rate, add_temperature=0.5):
        # op_completed = ""
        if random.random() < mutation_rate:
            action = random.choices(
                ["add_rule", "rm_rule"],
                weights=[add_temperature, 1 - add_temperature],
            )[0]
            rules = lsystem.rules
            rule_to_change = random.choice(range(0, len(rules)))
            rl_tmp = list(rules.values())[rule_to_change]
            splitted_rules = rl_tmp.split()
            gene_to_change = random.choice(range(0, len(splitted_rules)))
            match action:
                case "add_rule":
                    operator = random.choice([
                        "addf",
                        "addk",
                        "addl",
                        "addr",
                        "addb",
                        "addt",
                        "movf",
                        "movk",
                        "movl",
                        "movr",
                        "movt",
                        "movb",
                    ])
                    if operator in [
                        "addf",
                        "addk",
                        "addl",
                        "addr",
                        "addb",
                        "addt",
                    ]:
                        if splitted_rules[gene_to_change][:4] in [
                            "addf",
                            "addk",
                            "addl",
                            "addr",
                            "addb",
                            "addt",
                        ]:
                            rotation = random.choice([
                                0,
                                45,
                                90,
                                135,
                                180,
                                225,
                                270,
                            ])
                            op_to_add = operator + "(" + str(rotation) + ")"
                            item_to_add = random.choice(["B", "H", "N"])
                            splitted_rules.insert(
                                gene_to_change + 2, item_to_add
                            )
                            splitted_rules.insert(gene_to_change + 2, op_to_add)
                            # op_completed="ADDED : "+op_to_add+" "+item_to_add
                        elif splitted_rules[gene_to_change][:4] in [
                            "movf",
                            "movk",
                            "movl",
                            "movr",
                            "movb",
                            "movt",
                        ]:
                            rotation = random.choice([
                                0,
                                45,
                                90,
                                135,
                                180,
                                225,
                                270,
                            ])
                            op_to_add = operator + "(" + str(rotation) + ")"
                            item_to_add = random.choice(["B", "H", "N"])
                            splitted_rules.insert(gene_to_change, item_to_add)
                            splitted_rules.insert(gene_to_change, op_to_add)
                            # op_completed="ADDED : "+op_to_add+" "+item_to_add
                        elif splitted_rules[gene_to_change] in [
                            "C",
                            "B",
                            "H",
                            "N",
                        ]:
                            rotation = random.choice([
                                0,
                                45,
                                90,
                                135,
                                180,
                                225,
                                270,
                            ])
                            op_to_add = operator + "(" + str(rotation) + ")"
                            item_to_add = random.choice(["B", "H", "N"])
                            splitted_rules.insert(
                                gene_to_change + 1, item_to_add
                            )
                            splitted_rules.insert(gene_to_change + 1, op_to_add)
                            # op_completed="ADDED : "+op_to_add+" "+item_to_add
                    if operator in [
                        "movf",
                        "movk",
                        "movl",
                        "movr",
                        "movb",
                        "movt",
                    ]:
                        if splitted_rules[gene_to_change][:4] in [
                            "addf",
                            "addk",
                            "addl",
                            "addr",
                            "addb",
                            "addt",
                        ]:
                            splitted_rules.insert(gene_to_change + 2, operator)
                            # op_completed="ADDED : "+operator
                        elif splitted_rules[gene_to_change][:4] in [
                            "movf",
                            "movk",
                            "movl",
                            "movr",
                            "movb",
                            "movt",
                        ]:
                            splitted_rules.insert(gene_to_change, operator)
                            # op_completed="ADDED : "+operator
                        elif splitted_rules[gene_to_change] in [
                            "C",
                            "B",
                            "H",
                            "N",
                        ]:
                            splitted_rules.insert(gene_to_change + 1, operator)
                            # op_completed="ADDED : "+operator
                case "rm_rule":
                    if splitted_rules[gene_to_change][:4] in [
                        "addf",
                        "addk",
                        "addl",
                        "addr",
                        "addb",
                        "addt",
                    ]:
                        # op_completed="REMOVED : "+splitted_rules[gene_to_change]+" "+splitted_rules[gene_to_change+1]
                        splitted_rules.pop(gene_to_change)
                        splitted_rules.pop(gene_to_change)
                    elif splitted_rules[gene_to_change] in ["H", "B", "N"]:
                        # op_completed="REMOVED : "+splitted_rules[gene_to_change-1]+" "+splitted_rules[gene_to_change]
                        if gene_to_change - 1 >= 0:
                            splitted_rules.pop(gene_to_change - 1)
                            splitted_rules.pop(gene_to_change - 1)
                    elif splitted_rules[gene_to_change][:4] in [
                        "movf",
                        "movk",
                        "movl",
                        "movr",
                        "movt",
                        "movb",
                    ]:
                        # op_completed="REMOVED : "+splitted_rules[gene_to_change]
                        splitted_rules.pop(gene_to_change)
            new_rule = ""
            for j in range(0, len(splitted_rules)):
                new_rule += splitted_rules[j] + " "
            if new_rule != "":
                lsystem.rules[list(rules.keys())[rule_to_change]] = new_rule
            else:
                lsystem.rules[list(rules.keys())[rule_to_change]] = (
                    lsystem.rules[list(rules.keys())[rule_to_change]]
                )
        return lsystem


class CPPNMutator(Mutation):
    def __init__(self, next_innov_id_getter, next_node_id_getter):
        self.mutate = partial(
            self._mutate,
            next_innov_id_getter=next_innov_id_getter,
            next_node_id_getter=next_node_id_getter,
        )
        self.mutations_mapping = {"mutate_cppn": self.mutate}

    @staticmethod
    def _mutate(
        individual: Genotype,
        node_add_rate: float,
        conn_add_rate: float,
        node_remove_rate: float,
        conn_remove_rate: float,
        weight_mutate_rate: float,
        weight_mutate_power: float,
        next_innov_id_getter,  # function to get/update global innovation ID
        next_node_id_getter,  # function to get/update global node ID
    ):
        """
        Applies structural mutations (add/remove node/connection) and weight mutations.
        All mutations are applied sequentially based on their rates.
        """

        # 1. Add Connection
        if random.random() < conn_add_rate:
            individual = CPPNMutator._mutate_add_connection(
                individual, next_innov_id_getter
            )

        # 2. Add Node
        if random.random() < node_add_rate:
            individual = CPPNMutator._mutate_add_node(
                individual, next_innov_id_getter, next_node_id_getter
            )

        # 3. Remove Connection
        if random.random() < conn_remove_rate:
            individual = CPPNMutator._mutate_remove_connection(individual)

        # 4. Remove Node
        if random.random() < node_remove_rate:
            individual = CPPNMutator._mutate_remove_node(individual)

        # 5. Modify Weight
        if random.random() < weight_mutate_rate:
            individual = CPPNMutator._mutate_weight(
                individual, weight_mutate_power
            )

        return individual

    @staticmethod
    def _mutate_add_connection(individual: Genotype, next_innov_id_getter):
        """Try to add exactly one new acyclic connection."""
        if len(individual.nodes) < 2:
            return individual

        try:
            order = individual.get_node_ordering()
        except Exception:
            return individual

        rank = {nid: i for i, nid in enumerate(order)}

        # Attempt to find a valid pair
        node_ids = list(individual.nodes.keys())
        # Try a few times to find a valid pair to avoid infinite loops in dense graphs
        for _ in range(20):
            a, b = random.sample(node_ids, 2)
            na, nb = individual.nodes[a], individual.nodes[b]

            candidate = None
            if na.typ != "output" and nb.typ != "input" and rank[a] < rank[b]:
                candidate = (a, b)
            elif nb.typ != "output" and na.typ != "input" and rank[b] < rank[a]:
                candidate = (b, a)

            if not candidate:
                continue

            in_id, out_id = candidate

            # Check if connection exists
            exists = False
            for c in individual.connections.values():
                if c.in_id == in_id and c.out_id == out_id:
                    # If it exists but is disabled, we could re-enable it,
                    # but standard NEAT usually treats "add" as new structure.
                    # Here we just skip.
                    exists = True
                    break

            if exists:
                continue

            # Add new connection
            innov = next_innov_id_getter()
            weight = individual._get_random_weight()
            conn = Connection(
                in_id, out_id, weight, enabled=True, innov_id=innov
            )

            try:
                individual.add_connection(conn)
                return individual
            except ValueError:
                continue

        return individual

    @staticmethod
    def _mutate_add_node(
        individual: Genotype, next_innov_id_getter, next_node_id_getter
    ):
        """Split an enabled connection with a new hidden node."""
        enabled = [c for c in individual.connections.values() if c.enabled]
        if not enabled:
            return individual

        conn_to_split: Connection = random.choice(enabled)
        conn_to_split.enabled = False

        new_node_id = next_node_id_getter()
        new_node = Node(
            _id=new_node_id,
            typ="hidden",
            activation=individual._get_random_activation(),
            bias=individual._get_random_bias(),
        )
        individual.add_node(new_node)

        # in -> new
        innov1 = next_innov_id_getter()
        c1 = Connection(conn_to_split.in_id, new_node_id, 1.0, True, innov1)
        individual.add_connection(c1)

        # new -> out
        innov2 = next_innov_id_getter()
        c2 = Connection(
            new_node_id,
            conn_to_split.out_id,
            conn_to_split.weight,
            True,
            innov2,
        )
        individual.add_connection(c2)

        return individual

    @staticmethod
    def _mutate_weight(individual: Genotype, power: float):
        """Randomly select one connection and perturb its weight."""
        if not individual.connections:
            return individual

        # Pick a random connection (enabled or disabled, usually mutations can happen on disabled too,
        # but typically affect phenotype only if enabled. We mutate any to preserve genetic drift).
        conn = random.choice(list(individual.connections.values()))

        # Perturb weight
        delta = random.gauss(0, power)
        conn.weight += delta

        return individual

    @staticmethod
    def _mutate_remove_connection(individual: Genotype):
        """Randomly remove one existing connection."""
        if not individual.connections:
            return individual

        # Get list of innovation IDs
        innov_ids = list(individual.connections.keys())
        choice_id = random.choice(innov_ids)

        del individual.connections[choice_id]

        return individual

    @staticmethod
    def _mutate_remove_node(individual: Genotype):
        """
        Randomly remove one hidden node and all its attached connections.
        Cannot remove input or output nodes.
        """
        # Identify hidden nodes
        hidden_nodes = [
            nid
            for nid, node in individual.nodes.items()
            if node.typ == "hidden"
        ]

        if not hidden_nodes:
            return individual

        node_to_remove_id = random.choice(hidden_nodes)

        # 1. Remove the node
        del individual.nodes[node_to_remove_id]

        # 2. Remove all connections attached to this node
        # We must collect keys first to avoid "dictionary changed size during iteration"
        conns_to_remove = []
        for innov_id, conn in individual.connections.items():
            if (
                conn.in_id == node_to_remove_id
                or conn.out_id == node_to_remove_id
            ):
                conns_to_remove.append(innov_id)

        for innov_id in conns_to_remove:
            del individual.connections[innov_id]

        return individual


def test() -> None:
    """Entry point."""
    console.log(IntegersGenerator.integers(-5, 5, 5))
    example = IntegersGenerator.choice([1, 3, 4], (2, 5))
    console.log(example)
    example2 = IntegerMutator.integer_creep(
        example,
        span=1,
        mutation_probability=1,
    )
    console.log(example2)

    console.rule("[bold blue]Tree Generator Examples")

    treeGenerator = TreeGenerator()
    random_tree = treeGenerator.random_tree(max_depth=3, branching_prob=0.7)
    console.log("Random Tree:", random_tree)

    genome = TreeGenome()
    genome.root = TreeNode(
        pheno_config.ModuleInstance(
            type=pheno_config.ModuleType.BRICK,
            rotation=pheno_config.ModuleRotationsIdx.DEG_90,
            links={},
        )
    )
    genome.root.front = TreeNode(
        pheno_config.ModuleInstance(
            type=pheno_config.ModuleType.BRICK,
            rotation=pheno_config.ModuleRotationsIdx.DEG_45,
            links={},
        )
    )
    genome.root.left = TreeNode(
        pheno_config.ModuleInstance(
            type=pheno_config.ModuleType.BRICK,
            rotation=pheno_config.ModuleRotationsIdx.DEG_45,
            links={},
        )
    )
    tree_mutator = TreeMutator()
    mutated_genome = tree_mutator.random_subtree_replacement(
        genome, max_subtree_depth=1
    )
    console.log("Original Genome:", genome)
    console.log("Mutated Genome:", mutated_genome)


if __name__ == "__main__":
    test()
