"""
Experimental CPPN decoder, that isn't working yet.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from rich.console import Console
import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
    NUM_OF_TYPES_OF_MODULES,
)

if TYPE_CHECKING:
    from ariel.ec.genotypes.cppn.cppn_genome import CPPN_genotype
console = Console()

def softmax(raw_scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    e_x = np.exp(raw_scores - np.max(raw_scores))
    return e_x / e_x.sum()

class MorphologyDecoderBFS:
    """Decodes a CPPN using BFS with an adjustable 'core_bias'."""
    def __init__(self, cppn_genome: CPPN_genotype, max_modules: int = 20, core_bias: float = 0.0):
        self.cppn_genome = cppn_genome
        self.max_modules = max_modules
        self.core_bias = core_bias # New parameter to prioritize the core
        self.face_deltas = {
            ModuleFaces.FRONT: (1, 0, 0), ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0), ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1), ModuleFaces.LEFT: (0, 0, -1),
        }

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (parent_pos[0] + delta[0], parent_pos[1] + delta[1], parent_pos[2] + delta[2])

    def decode(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        module_data = {}
        core_id, core_pos, core_type, core_rot = IDX_OF_CORE, (0, 0, 0), ModuleType.CORE, ModuleRotationsIdx.DEG_0
        robot_graph.add_node(core_id, type=core_type.name, rotation=core_rot.name)
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {'pos': core_pos, 'type': core_type, 'rot': core_rot}
        next_module_id = 1
        current_layer = [core_id]

        while current_layer:
            next_layer = []
            
            for parent_id in current_layer:
                if len(robot_graph) >= self.max_modules:
                    break

                potential_children = []
                parent_pos = module_data[parent_id]['pos']
                parent_type = module_data[parent_id]['type']

                for face in ModuleFaces:
                    if face not in ALLOWED_FACES[parent_type]:
                        continue
                    
                    child_pos = self._get_child_coords(parent_pos, face)
                    
                    if child_pos in occupied_coords:
                        continue

                    cppn_inputs = list(parent_pos) + list(child_pos)
                    raw_outputs = self.cppn_genome.activate(cppn_inputs)
                    
                    conn_score = raw_outputs[0]

                    # this is where I add a bias towards connecting to the core 
                    if parent_id == IDX_OF_CORE:
                        conn_score += self.core_bias

                    type_scores = np.array(raw_outputs[1:1+NUM_OF_TYPES_OF_MODULES])
                    rot_scores = np.array(raw_outputs[1+NUM_OF_TYPES_OF_MODULES:])

                    child_type = ModuleType(np.argmax(softmax(type_scores)))
                    child_rot = ModuleRotationsIdx(np.argmax(softmax(rot_scores)))
                    
                    if child_type not in (ModuleType.NONE, ModuleType.CORE) and \
                       face in ALLOWED_FACES[child_type] and child_rot in ALLOWED_ROTATIONS[child_type]:
                        potential_children.append({
                            'score': conn_score, 'child_pos': child_pos,
                            'child_type': child_type, 'child_rot': child_rot, 'face': face,
                        })
                
                if potential_children:
                    best_child = max(potential_children, key=lambda x: x['score'])

                    if best_child['child_pos'] in occupied_coords:
                        continue
                    
                    child_id = next_module_id
                    robot_graph.add_node(child_id, type=best_child['child_type'].name, rotation=best_child['child_rot'].name)
                    robot_graph.add_edge(parent_id, child_id, face=best_child['face'].name)
                    
                    occupied_coords[best_child['child_pos']] = child_id
                    
                    module_data[child_id] = {'pos': best_child['child_pos'], 'type': best_child['child_type'], 'rot': best_child['child_rot']}
                    next_layer.append(child_id)
                    next_module_id += 1

            current_layer = next_layer

        return robot_graph
