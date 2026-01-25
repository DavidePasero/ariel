"""Neural developmental encoding.

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

from __future__ import annotations
import networkx as nx
import numpy as np
import json

from ariel.ec.genotypes.genotype import Genotype, MAX_MODULES
import torch
from pathlib import Path
from rich.console import Console
from rich.traceback import install
import base64, io, gzip


# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)

from ariel.ec.crossovers import NDECrossover
from ariel.ec.mutations import NDEMutation

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)

# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP ---
SEED = 42
NETWORK_SEED = 12345
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP ---
install(show_locals=True)
console = Console()


def set_run_seed(seed: int) -> None:
    global RUN_SEED, RNG
    RUN_SEED = int(seed)
    RNG = np.random.default_rng(RUN_SEED)


class NeuralDevelopmentalEncoding(torch.nn.Module):
    def __init__(self, number_of_modules: int) -> None:
        super().__init__()

        # ! ----------------------------------------------------------------- #
        # self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # ! ----------------------------------------------------------------- #

        # Hidden Layers
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 128)

        # ------------------------------------------------------------------- #
        # OUTPUTS
        self.type_p_shape = (number_of_modules, NUM_OF_TYPES_OF_MODULES)
        self.type_p_out = torch.nn.Linear(
            128,
            number_of_modules * NUM_OF_TYPES_OF_MODULES,
        )

        self.conn_p_shape = (number_of_modules, number_of_modules, NUM_OF_FACES)
        self.conn_p_out = torch.nn.Linear(
            128,
            number_of_modules * number_of_modules * NUM_OF_FACES,
        )

        self.rot_p_shape = (number_of_modules, NUM_OF_ROTATIONS)
        self.rot_p_out = torch.nn.Linear(
            128,
            number_of_modules * NUM_OF_ROTATIONS,
        )

        self.output_layers = [
            self.type_p_out,
            self.conn_p_out,
            self.rot_p_out,
        ]
        self.output_shapes = [
            self.type_p_shape,
            self.conn_p_shape,
            self.rot_p_shape,
        ]
        # ------------------------------------------------------------------- #

        # Activations
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        genotype: list[npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        outputs: list[npt.NDArray[np.float32]] = []
        for idx, chromosome in enumerate(genotype):
            with torch.no_grad():  # double safety
                x = torch.from_numpy(chromosome).to(torch.float32)

                x = self.fc1(x)
                x = self.relu(x)

                x = self.fc2(x)
                x = self.relu(x)

                x = self.fc3(x)
                x = self.relu(x)

                x = self.fc4(x)
                x = self.relu(x)

                x = self.output_layers[idx](x)
                x = self.sigmoid(x)

                x = x.view(self.output_shapes[idx])
                outputs.append(x.detach().numpy())
        return outputs

    @staticmethod
    def make_network_from_seed(
        seed: int, num_modules: int
    ) -> NeuralDevelopmentalEncoding:
        torch.manual_seed(seed)
        net = NeuralDevelopmentalEncoding(num_modules)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net


NETWORK_TEMPLATE = NeuralDevelopmentalEncoding.make_network_from_seed(
    NETWORK_SEED, MAX_MODULES
)


class NDEGenome(Genotype):
    def __init__(self, individual: np.array):
        self.individual = individual
        self.num_modules = MAX_MODULES

    @staticmethod
    def get_crossover_object() -> "Crossover":
        """Return the crossover operator for this genotype type."""
        return NDECrossover()

    @staticmethod
    def get_mutator_object() -> "Mutation":
        """Return the mutator operator for this genotype type."""
        return NDEMutation()

    @staticmethod
    def create_individual(**kwargs: dict) -> "Genotype":
        """Generate a new individual of this genotype type."""
        genotype_size = 64
        type_p_genes = RNG.random(genotype_size)
        conn_p_genes = RNG.random(genotype_size)
        rot_p_genes = RNG.random(genotype_size)

        genotype = [
            type_p_genes,
            conn_p_genes,
            rot_p_genes,
        ]
        return NDEGenome(np.array(genotype))

    def to_digraph(self: "Genotype", **kwargs: dict) -> nx.DiGraph:
        type_p, conn_p, rot_p = NETWORK_TEMPLATE.forward(self.individual)

        hpd = HighProbabilityDecoder(self.num_modules)
        return hpd.probability_matrices_to_graph(type_p, conn_p, rot_p)

    def to_json(self, **kwargs) -> str:
        ind_np = np.asarray(self.individual)
        payload = {
            "schema": "NDEGenome.v3",
            "num_modules": int(self.num_modules),
            "network_seed": int(NETWORK_SEED),
            "individual": {
                "dtype": str(ind_np.dtype),
                "shape": list(ind_np.shape),
                "data": ind_np.tolist(),
            },
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(json_data: str, load_global_network: bool = False, **kwargs):
        obj = json.loads(json_data)

        ind = obj["individual"]
        ind_np = np.array(ind["data"], dtype=np.dtype(ind["dtype"])).reshape(
            ind["shape"]
        )

        if load_global_network:
            seed = int(obj["network_seed"])
            global NETWORK_TEMPLATE
            NETWORK_TEMPLATE = (
                NeuralDevelopmentalEncoding.make_network_from_seed(
                    SEED, int(obj["num_modules"])
                )
            )

        return NDEGenome(ind_np)


if __name__ == "__main__":
    """Usage example."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=20)

    genotype_size = 64
    type_p_genes = RNG.random(genotype_size)
    conn_p_genes = RNG.random(genotype_size)
    rot_p_genes = RNG.random(genotype_size)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    outputs = nde.forward(genotype)
    for output in outputs:
        console.log(output.shape)
