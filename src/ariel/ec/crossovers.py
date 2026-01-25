"""TODO(jmdm): description of script."""

from __future__ import annotations

import random

# Standard library
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.ec.genotypes.cppn.cppn_genome import CPPN_genotype
from ariel.ec.genotypes.lsystem.l_system_genotype import LSystemDecoder

if TYPE_CHECKING:
    from ariel.ec.genotypes.cppn.connection import Connection
    from ariel.ec.genotypes.cppn.node import Node
    from ariel.ec.genotypes.genotype import Genotype
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome
    from ariel.ec.genotypes.nde.nde import NDEGenome


from ariel.ec.genotypes.genotype import MAX_MODULES

# Max attempts to find valid crossover points
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


class Crossover(ABC):
    crossovers_mapping: dict[str, function] = NotImplemented
    which_crossover: str = ""

    @classmethod
    def set_which_crossover(cls, crossover_type: str) -> None:
        if crossover_type not in cls.crossovers_mapping:
            msg = f"Crossover type '{crossover_type}' not recognized."
            raise ValueError(msg)
        cls.which_crossover = crossover_type

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.crossovers_mapping = {
            name: getattr(cls, name)
            for name, val in cls.__dict__.items()
            if isinstance(val, staticmethod) and not name.startswith("_")
        }

    def __call__(
        self,
        parent_i: Genotype,
        parent_j: Genotype,
        **kwargs: dict,
    ) -> tuple[Genotype, ...]:
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
        return self.crossovers_mapping[self.which_crossover](
            parent_i, parent_j, **kwargs
        )


class IntegerCrossover(Crossover):
    @staticmethod
    def one_point(
        parent_i: Genotype,
        parent_j: Genotype,
    ) -> tuple[Genotype, Genotype]:
        # Prep
        parent_i_arr_shape = np.array(parent_i).shape
        parent_j_arr_shape = np.array(parent_j).shape
        parent_i_arr = np.array(parent_i).flatten().copy()
        parent_j_arr = np.array(parent_j).flatten().copy()

        # Ensure parents have the same length
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)

        # Select crossover point
        crossover_point = RNG.integers(0, len(parent_i_arr))

        # Copy over parents
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()

        # Perform crossover
        child1[crossover_point:] = parent_j_arr[crossover_point:]
        child2[crossover_point:] = parent_i_arr[crossover_point:]

        # Correct final shape
        child1 = child1.reshape(parent_i_arr_shape).astype(int).tolist()
        child2 = child2.reshape(parent_j_arr_shape).astype(int).tolist()
        return child1, child2


class TreeCrossover(Crossover):
    @staticmethod
    def koza_default(
        parent_i: TreeGenome,
        parent_j: TreeGenome,
        koza_internal_node_prob: float = 0.9,
    ) -> tuple[TreeGenome, TreeGenome]:
        """
        Koza default:
            -   In Parent A: choose an internal node with high probability (e.g., 90%) excluding root.
                Falls back to any node if A has no internal nodes.
            -   In Parent B: choose any node uniformly (internal or terminal).

        Forcing at least one internal node increases the chance you actually change structure
        (not just swapping a leaf for a leaf), while letting the other parent be unrestricted adds variety.
        """
        parent_i_root, parent_j_root = parent_i.root, parent_j.root

        nodes_a = parent_i_root.get_all_nodes(exclude_root=True)
        nodes_b = parent_j_root.get_all_nodes(exclude_root=True)

        # If either tree is just a root, return copies of parents
        if not nodes_a or not nodes_b:
            return parent_i.copy(), parent_j.copy()

        parent_i_internal_nodes = parent_i_root.get_internal_nodes(
            mode="dfs", exclude_root=True
        )

        if RNG.random() > koza_internal_node_prob and parent_i_internal_nodes:
            node_a = RNG.choice(parent_i_internal_nodes)
        else:
            node_a = RNG.choice(
                parent_i_root.get_all_nodes(mode="dfs", exclude_root=True)
            )

        parent_j_all_nodes = parent_j_root.get_all_nodes()
        node_b = RNG.choice(parent_j_all_nodes)
        if node_a is None or node_b is None:
            # If either tree is just a root, return copies of parents
            return parent_i.copy(), parent_j.copy()

        parent_i_old = parent_i.copy()
        parent_j_old = parent_j.copy()
        child1 = parent_i
        child2 = parent_j

        child1.root.replace_node(node_a, node_b)
        child2.root.replace_node(node_b, node_a)

        parent_i = parent_i_old
        parent_j = parent_j_old
        return child1, child2

    @staticmethod
    def normal(
        parent_i: TreeGenome,
        parent_j: TreeGenome,
        koza_internal_node_prob: float = 0.9,
    ) -> tuple[TreeGenome, TreeGenome]:
        """
        Normal tree crossover:
            - Pick a random node from Parent i (uniform over all nodes).
            - Pick a random node from Parent j (uniform over all nodes).
            - Swap the selected subtrees.

        Returns two children produced by swapping the chosen subtrees.
        """
        parent_i_root, parent_j_root = parent_i.root, parent_j.root

        nodes_i = parent_i_root.get_all_nodes(exclude_root=True)
        nodes_j = parent_j_root.get_all_nodes(exclude_root=True)

        # If either tree is just a root, return copies of parents
        if not nodes_i or not nodes_j:
            return parent_i.copy(), parent_j.copy()

        # Uniformly choose any node (root, internal, or leaf)
        node_i = RNG.choice(nodes_i)
        node_j = RNG.choice(nodes_j)

        # With the crossover we don't want to exceed MAX_MODULES
        i = 0
        while (
            parent_i.num_modules
            - node_i.num_descendants
            + node_j.num_descendants
            > MAX_MODULES
            or parent_j.num_modules
            - node_j.num_descendants
            + node_i.num_descendants
            > MAX_MODULES
        ):
            node_i = RNG.choice(nodes_i)
            node_j = RNG.choice(nodes_j)
            i += 1
            if i >= MAX_ATTEMPTS:
                return parent_i.copy(), parent_j.copy()

        # Preserve originals (same pattern as in koza_default)
        parent_i_old = parent_i.copy()
        parent_j_old = parent_j.copy()
        child1 = parent_i
        child2 = parent_j

        # Perform the swap
        child1.root.replace_node(node_i, node_j)
        child2.root.replace_node(node_j, node_i)

        # Restore parent handles for caller (as in your koza_default)
        parent_i = parent_i_old
        parent_j = parent_j_old
        return child1, child2


class LSystemCrossover(Crossover):
    @staticmethod
    def crossover_uniform_rules_lsystem(
        lsystem_parent1, lsystem_parent2, mutation_rate
    ):
        axiom_offspring1 = "C"
        axiom_offspring2 = "C"
        rules_offspring1 = {}
        rules_offspring2 = {}
        iter_offspring1 = 0
        iter_offspring2 = 0
        if random.random() > mutation_rate:
            rules_offspring1["C"] = lsystem_parent2.rules["C"]
            rules_offspring2["C"] = lsystem_parent1.rules["C"]
            iter_offspring1 += lsystem_parent2.iterations
            iter_offspring2 += lsystem_parent1.iterations
        else:
            rules_offspring1["C"] = lsystem_parent1.rules["C"]
            rules_offspring2["C"] = lsystem_parent2.rules["C"]
            iter_offspring1 += lsystem_parent1.iterations
            iter_offspring2 += lsystem_parent2.iterations
        if random.random() > mutation_rate:
            rules_offspring1["B"] = lsystem_parent2.rules["B"]
            rules_offspring2["B"] = lsystem_parent1.rules["B"]
            iter_offspring1 += lsystem_parent2.iterations
            iter_offspring2 += lsystem_parent1.iterations
        else:
            rules_offspring1["B"] = lsystem_parent1.rules["B"]
            rules_offspring2["B"] = lsystem_parent2.rules["B"]
            iter_offspring1 += lsystem_parent1.iterations
            iter_offspring2 += lsystem_parent2.iterations
        if random.random() > mutation_rate:
            rules_offspring1["H"] = lsystem_parent2.rules["H"]
            rules_offspring2["H"] = lsystem_parent1.rules["H"]
            iter_offspring1 += lsystem_parent2.iterations
            iter_offspring2 += lsystem_parent1.iterations
        else:
            rules_offspring1["H"] = lsystem_parent1.rules["H"]
            rules_offspring2["H"] = lsystem_parent2.rules["H"]
            iter_offspring1 += lsystem_parent1.iterations
            iter_offspring2 += lsystem_parent2.iterations
        if random.random() > mutation_rate:
            rules_offspring1["N"] = lsystem_parent2.rules["N"]
            rules_offspring2["N"] = lsystem_parent1.rules["N"]
            iter_offspring1 += lsystem_parent2.iterations
            iter_offspring2 += lsystem_parent1.iterations
        else:
            rules_offspring1["N"] = lsystem_parent1.rules["N"]
            rules_offspring2["N"] = lsystem_parent2.rules["N"]
            iter_offspring1 += lsystem_parent1.iterations
            iter_offspring2 += lsystem_parent2.iterations
        iteration_offspring1 = int(iter_offspring1 / 4)
        iteration_offspring2 = int(iter_offspring2 / 4)
        offspring1 = LSystemDecoder(
            axiom_offspring1,
            rules_offspring1,
            iteration_offspring1,
            lsystem_parent1.max_elements,
            lsystem_parent1.max_depth,
            lsystem_parent1.verbose,
        )
        offspring2 = LSystemDecoder(
            axiom_offspring2,
            rules_offspring2,
            iteration_offspring2,
            lsystem_parent2.max_elements,
            lsystem_parent2.max_depth,
            lsystem_parent2.verbose,
        )
        return offspring1, offspring2

    @staticmethod
    def crossover_uniform_genes_lsystem(
        lsystem_parent1, lsystem_parent2, mutation_rate
    ):
        axiom_offspring1 = "C"
        axiom_offspring2 = "C"
        rules_offspring1 = {}
        rules_offspring2 = {}
        iter_offspring1 = 0
        iter_offspring2 = 0

        rules_parent1 = lsystem_parent1.rules["C"].split()
        rules_parent2 = lsystem_parent2.rules["C"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent1[i] + " " + rules_parent1[i + 1]
                enh_parent1.append(new_token)
                i += 1
            elif rules_parent1[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent1.append(rules_parent1[i])
            elif rules_parent1[i] == "C":
                enh_parent1.append(rules_parent1[i])
            i += 1
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent2[i] + " " + rules_parent2[i + 1]
                enh_parent2.append(new_token)
                i += 1
            elif rules_parent2[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent2.append(rules_parent2[i])
            elif rules_parent2[i] == "C":
                enh_parent2.append(rules_parent2[i])
            i += 1
        r_offspring1 = ""
        r_offspring2 = ""
        le_common = min(len(enh_parent1), len(enh_parent2))
        for i in range(0, le_common):
            if random.random() > mutation_rate:
                r_offspring1 += enh_parent2[i] + " "
                r_offspring2 += enh_parent1[i] + " "
            else:
                r_offspring1 += enh_parent1[i] + " "
                r_offspring2 += enh_parent2[i] + " "
        if len(enh_parent1) > le_common:
            for j in range(le_common, len(enh_parent1)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent1[j] + " "
                else:
                    r_offspring2 += enh_parent1[j] + " "
        if len(enh_parent2) > le_common:
            for j in range(le_common, len(enh_parent2)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent2[j] + " "
                else:
                    r_offspring2 += enh_parent2[j] + " "
        rules_offspring1["C"] = r_offspring1
        rules_offspring2["C"] = r_offspring2

        rules_parent1 = lsystem_parent1.rules["B"].split()
        rules_parent2 = lsystem_parent2.rules["B"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent1[i] + " " + rules_parent1[i + 1]
                enh_parent1.append(new_token)
                i += 1
            elif rules_parent1[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent1.append(rules_parent1[i])
            elif rules_parent1[i] == "B":
                enh_parent1.append(rules_parent1[i])
            i += 1
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent2[i] + " " + rules_parent2[i + 1]
                enh_parent2.append(new_token)
                i += 1
            elif rules_parent2[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent2.append(rules_parent2[i])
            elif rules_parent2[i] == "B":
                enh_parent2.append(rules_parent2[i])
            i += 1
        r_offspring1 = ""
        r_offspring2 = ""
        le_common = min(len(enh_parent1), len(enh_parent2))
        for i in range(0, le_common):
            if random.random() > mutation_rate:
                r_offspring1 += enh_parent2[i] + " "
                r_offspring2 += enh_parent1[i] + " "
            else:
                r_offspring1 += enh_parent1[i] + " "
                r_offspring2 += enh_parent2[i] + " "
        if len(enh_parent1) > le_common:
            for j in range(le_common, len(enh_parent1)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent1[j] + " "
                else:
                    r_offspring2 += enh_parent1[j] + " "
        if len(enh_parent2) > le_common:
            for j in range(le_common, len(enh_parent2)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent2[j] + " "
                else:
                    r_offspring2 += enh_parent2[j] + " "
        rules_offspring1["B"] = r_offspring1
        rules_offspring2["B"] = r_offspring2

        rules_parent1 = lsystem_parent1.rules["H"].split()
        rules_parent2 = lsystem_parent2.rules["H"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent1[i] + " " + rules_parent1[i + 1]
                enh_parent1.append(new_token)
                i += 1
            elif rules_parent1[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent1.append(rules_parent1[i])
            elif rules_parent1[i] == "H":
                enh_parent1.append(rules_parent1[i])
            i += 1
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent2[i] + " " + rules_parent2[i + 1]
                enh_parent2.append(new_token)
                i += 1
            elif rules_parent2[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent2.append(rules_parent2[i])
            elif rules_parent2[i] == "H":
                enh_parent2.append(rules_parent2[i])
            i += 1
        r_offspring1 = ""
        r_offspring2 = ""
        le_common = min(len(enh_parent1), len(enh_parent2))
        for i in range(0, le_common):
            if random.random() > mutation_rate:
                r_offspring1 += enh_parent2[i] + " "
                r_offspring2 += enh_parent1[i] + " "
            else:
                r_offspring1 += enh_parent1[i] + " "
                r_offspring2 += enh_parent2[i] + " "
        if len(enh_parent1) > le_common:
            for j in range(le_common, len(enh_parent1)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent1[j] + " "
                else:
                    r_offspring2 += enh_parent1[j] + " "
        if len(enh_parent2) > le_common:
            for j in range(le_common, len(enh_parent2)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent2[j] + " "
                else:
                    r_offspring2 += enh_parent2[j] + " "
        rules_offspring1["H"] = r_offspring1
        rules_offspring2["H"] = r_offspring2

        rules_parent1 = lsystem_parent1.rules["N"].split()
        rules_parent2 = lsystem_parent2.rules["N"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent1[i] + " " + rules_parent1[i + 1]
                enh_parent1.append(new_token)
                i += 1
            elif rules_parent1[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent1.append(rules_parent1[i])
            elif rules_parent1[i] == "N":
                enh_parent1.append(rules_parent1[i])
            i += 1
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in [
                "addf",
                "addk",
                "addl",
                "addr",
                "addb",
                "addt",
            ]:
                new_token = rules_parent2[i] + " " + rules_parent2[i + 1]
                enh_parent2.append(new_token)
                i += 1
            elif rules_parent2[i][:4] in [
                "movf",
                "movk",
                "movl",
                "movr",
                "movb",
                "movt",
            ]:
                enh_parent2.append(rules_parent2[i])
            elif rules_parent2[i] == "N":
                enh_parent2.append(rules_parent2[i])
            i += 1
        r_offspring1 = ""
        r_offspring2 = ""
        le_common = min(len(enh_parent1), len(enh_parent2))
        for i in range(0, le_common):
            if random.random() > mutation_rate:
                r_offspring1 += enh_parent2[i] + " "
                r_offspring2 += enh_parent1[i] + " "
            else:
                r_offspring1 += enh_parent1[i] + " "
                r_offspring2 += enh_parent2[i] + " "
        if len(enh_parent1) > le_common:
            for j in range(le_common, len(enh_parent1)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent1[j] + " "
                else:
                    r_offspring2 += enh_parent1[j] + " "
        if len(enh_parent2) > le_common:
            for j in range(le_common, len(enh_parent2)):
                if random.random() > mutation_rate:
                    r_offspring1 += enh_parent2[j] + " "
                else:
                    r_offspring2 += enh_parent2[j] + " "
        rules_offspring1["N"] = r_offspring1
        rules_offspring2["N"] = r_offspring2

        iter_offspring1 += lsystem_parent2.iterations
        iter_offspring2 += lsystem_parent1.iterations
        offspring1 = LSystemDecoder(
            axiom_offspring1,
            rules_offspring1,
            iter_offspring1,
            lsystem_parent1.max_elements,
            lsystem_parent1.max_depth,
            lsystem_parent1.verbose,
        )
        offspring2 = LSystemDecoder(
            axiom_offspring2,
            rules_offspring2,
            iter_offspring2,
            lsystem_parent2.max_elements,
            lsystem_parent2.max_depth,
            lsystem_parent2.verbose,
        )
        return offspring1, offspring2


class CPPNCrossover(Crossover):
    @staticmethod
    def crossover(
        parent_i: CPPN_genotype, parent_j: CPPN_genotype
    ) -> tuple[CPPN_genotype, CPPN_genotype]:
        # Select fitter
        if parent_i.fitness >= parent_j.fitness:
            fitter, other = parent_i, parent_j
        else:
            fitter, other = parent_j, parent_i

        # If equal fitness and length differs, standard NEAT prefers NOT the shorter as fitter
        # but we’ll keep your tie-break if you need it.

        # Start offspring with no genes; we’ll add safely
        nodes: dict[int, Node] = {}
        conns: dict[int, Connection] = {}

        child = CPPN_genotype(nodes, conns, fitness=0.0)

        # Bring in any node that might be referenced later (copy lazily when needed)
        def ensure_node(nid: int, source: CPPN_genotype):
            if nid not in child.nodes:
                child.nodes[nid] = source.nodes[nid].copy()

        # Helper: attempt to add a connection safely
        def try_add_copy(src_conn: Connection, from_parent: CPPN_genotype):
            ensure_node(src_conn.in_id, from_parent)
            ensure_node(src_conn.out_id, from_parent)
            # preserve enabled status using NEAT's rule:
            # If a matching gene exists in both parents and is disabled in either, keep it disabled
            enabled = src_conn.enabled
            other_conn = other.connections.get(src_conn.innov_id)
            if other_conn and (not src_conn.enabled or not other_conn.enabled):
                enabled = False

            # Clone with the (possibly) updated enabled bit
            c = src_conn.copy()
            c.enabled = enabled

            # Only add if enabled or (optional) you want to store disabled too.
            # If you keep disabled edges, they must NOT be considered in cycle checks for execution.
            try:
                if c.enabled:
                    child.add_connection(c)  # will reject cycles/illegal edges
                else:
                    # Keep disabled edges as metadata without affecting topology:
                    child.connections[c.innov_id] = c
            except ValueError:
                # Edge would create a cycle or illegal direction → skip it
                pass

        # 1) Add fitter parent’s connections first
        for conn in fitter.connections.values():
            try_add_copy(conn, fitter)

        # 2) Merge the other parent’s connections per NEAT rules
        for innov_id, conn_b in other.connections.items():
            conn_a = fitter.connections.get(innov_id)
            if conn_a:
                # matching gene → randomly choose (weight etc.), then apply disabled rule
                chosen = random.choice([conn_a, conn_b]).copy()
                try_add_copy(chosen, fitter if chosen is conn_a else other)
            else:
                # disjoint/excess from less fit → NEAT: inherit only if equal fitness
                if other.fitness == fitter.fitness:
                    try_add_copy(conn_b, other)

        # Offspring 1 unchanged
        return parent_i.copy(), child


class NDECrossover(Crossover):
    @staticmethod
    def one_point(
        parent_i: NDEGenome,
        parent_j: NDEGenome,
        **kwargs,
    ) -> tuple[NDEGenome, NDEGenome]:
        from ariel.ec.genotypes.nde.nde import NDEGenome

        # Prep
        parent_i_arr_shape = np.array(parent_i.individual).shape
        parent_j_arr_shape = np.array(parent_j.individual).shape
        parent_i_arr = np.array(parent_i.individual).flatten().copy()
        parent_j_arr = np.array(parent_j.individual).flatten().copy()

        # Ensure parents have the same length
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)

        # Select crossover point
        crossover_point = RNG.integers(0, len(parent_i_arr))

        # Copy over parents
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()

        # Perform crossover
        child1[crossover_point:] = parent_j_arr[crossover_point:]
        child2[crossover_point:] = parent_i_arr[crossover_point:]

        # Correct final shape
        child1 = NDEGenome(individual=child1.reshape(parent_i_arr_shape))
        child2 = NDEGenome(individual=child2.reshape(parent_j_arr_shape))
        return child1, child2

    @staticmethod
    def revde(
        parent_i: NDEGenome,
        parent_j: NDEGenome,
        parent_k: NDEGenome,
        scaling_factor: float,
        **kwargs,
    ) -> tuple[NDEGenome, NDEGenome, NDEGenome]:
        from ariel.ec.genotypes.nde.nde import NDEGenome

        original_shape = parent_i.individual.shape

        parent_i_genome = parent_i.individual.flatten()
        parent_j_genome = parent_j.individual.flatten()
        parent_k_genome = parent_k.individual.flatten()

        # Passed parameters
        f = scaling_factor
        f2 = f**2
        f3 = f**3

        # Prep work
        a = 1 - f2
        b = f + f2
        c = -f + f2 + f3
        d = 1 - (2 * f2) - f3

        # Linear transformation matrix
        r_matrix = np.array([
            [1, f, -f],
            [-f, a, b],
            [b, c, d],
        ])

        # Ensure parents have the same shape
        if not (
            parent_i_genome.shape
            == parent_j_genome.shape
            == parent_k_genome.shape
        ):
            msg = "Parents must have the same shape"
            raise ValueError(msg)

        # Perform mutation
        # 1. Stack parents to shape (3, 3, n_genes)
        # 2. Reshape to (3, N) where N is total genes (flat) -> (3, 3*n_genes)
        x_matrix = np.stack((
            parent_i_genome,
            parent_j_genome,
            parent_k_genome,
        ))

        # 3. Matrix Multiplication: (3,3) @ (3, 3*n_genes) -> (3, 3*n_genes)
        out = r_matrix @ x_matrix

        # 4. Reshape back to original dimensions and unpack
        y1 = out[0].reshape(original_shape)
        y2 = out[1].reshape(original_shape)
        y3 = out[2].reshape(original_shape)

        child_i = NDEGenome(y1)
        child_j = NDEGenome(y2)
        child_k = NDEGenome(y3)
        return child_i, child_j, child_k


def tree_main():
    import ariel.body_phenotypes.robogen_lite.config as config
    from ariel.ec.genotypes.tree.tree_genome import TreeNode, TreeGenome

    # Create first tree
    genome1 = TreeGenome()
    genome1.root = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.CORE,
            rotation=config.ModuleRotationsIdx.DEG_0,
            links={},
        )
    )
    genome1.root.front = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.BRICK,
            rotation=config.ModuleRotationsIdx.DEG_90,
            links={},
        )
    )
    genome1.root.back = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.HINGE,
            rotation=config.ModuleRotationsIdx.DEG_45,
            links={},
        )
    )

    # Create second tree
    genome2 = TreeGenome()
    genome2.root = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.CORE,
            rotation=config.ModuleRotationsIdx.DEG_0,
            links={},
        )
    )
    genome2.root.right = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.BRICK,
            rotation=config.ModuleRotationsIdx.DEG_180,
            links={},
        )
    )
    genome2.root.back = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.HINGE,
            rotation=config.ModuleRotationsIdx.DEG_270,
            links={},
        )
    )

    console.log("Parent 1:", genome1)
    console.log("Parent 2:", genome2)

    genome2.root.replace_node(genome1, genome2)

    # Perform crossover
    child1, child2 = TreeCrossover.koza_default(genome1, genome2)

    console.log("Child 1:", child1)
    console.log("Child 2:", child2)


def main() -> None:
    """Entry point."""
    p1 = IntegersGenerator.integers(-5, 5, (2, 5))
    p2 = IntegersGenerator.choice([1, 3, 4], (2, 5))
    console.log(p1, p2)

    c1, c2 = Crossover.one_point(p1, p2)
    console.log(c1, c2)


if __name__ == "__main__":
    main()
