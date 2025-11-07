from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import mujoco as mj
if TYPE_CHECKING:
    from ariel.ec.genotypes.cppn.cppn_genome import CPPN_genotype
class CppnControllerCallback:
    """
    A callable object that implements the controller_callback_function
    by querying a 'Genome' (CPPN) instance.

    This class is designed to be instantiated once and then have its
    genome updated by the evolutionary algorithm before each simulation.
    """
    
    def __init__(self,
                 motor_coordinates: np.ndarray,
                 time_frequency: float = 1.0,
                 bias_input: float = 1.0):
        """
        Initializes the CppnControllerCallback.

        Parameters
        ----------
        motor_coordinates : np.ndarray
            A (N_actuators, 3) numpy array where each row is the
            (x, y, z) coordinate for the corresponding motor.
        time_frequency : float, optional
            The frequency (omega) for the sin/cos time inputs.
        bias_input : float, optional
            A constant bias value to be fed into the CPPN.
        """
        self.motor_coords = motor_coordinates
        self.num_actuators = motor_coordinates.shape[0]
        self.freq = time_frequency
        self.bias = bias_input
        
        # The genome to be evaluated.
        # This will be set by the EA before simulation.
        self.genome: CPPN_genotype | None = None

        # Your Controller clips to [-pi/2, pi/2], so we will
        # scale our CPPN's output to this target range.
        self.target_min = -np.pi / 2
        self.target_max = np.pi / 2

    def set_genome(self, genome: CPPN_genotype) -> None:
        """
        Sets the active Genome (CPPN) for the next simulation.
        This is the main entry point for the evolutionary algorithm.
        """
        # ---
        # NOTE: Your Genome.random() creates inputs and outputs.
        # Your EA must ensure that the genomes being set here have
        # the correct number of inputs (e.g., 6) and outputs (e.g., 1).
        # ---
        
        # Check for expected number of outputs (should be 1)
        output_nodes = [n for n in genome.nodes.values() if n.typ == 'output']
        if len(output_nodes) != 1:
            raise ValueError(
                f"Genome must have exactly 1 output node, but found {len(output_nodes)}"
            )

        # (Optional) Check for expected number of inputs
        input_nodes = [n for n in genome.nodes.values() if n.typ == 'input']
        # We expect 6 inputs: x, y, z, t_sin, t_cos, bias
        if len(input_nodes) != 6: 
            print(f"Warning: Genome has {len(input_nodes)} inputs, but "
                  f"this callback will provide 6 (x, y, z, t_sin, t_cos, bias).")

        self.genome = genome

    def __call__(self, model: mj.MjModel, data: mj.MjData, *args, **kwargs) -> np.ndarray:
        """
        This is the `controller_callback_function`.
        It is called by the `Controller` class every `time_steps_per_ctrl_step`.
        """
        
        if self.genome is None:
            raise RuntimeError(
                "CPPN callback was called before a genome was set using 'set_genome()'."
            )

        # 1. Get temporal inputs
        t = data.time
        t_sine = np.sin(self.freq * t)
        t_cosine = np.cos(self.freq * t)
        
        # 2. Initialize output array
        target_positions = np.zeros(self.num_actuators)

        # 3. Query the CPPN for each motor
        for i in range(self.num_actuators):
            
            # 3a. Get spatial inputs for this motor
            coords = self.motor_coords[i]
            
            # 3b. Build the full input vector for the CPPN
            # (Order: x, y, z, t_sin, t_cos, bias)
            cppn_inputs = [
                coords[0], coords[1], coords[2],
                t_sine, t_cosine, self.bias
            ]

            # 3c. Activate the network (Genome)
            # This returns a list of outputs. We expect 1 output.
            cppn_output_list = self.genome.activate(cppn_inputs)
            
            # Use the first (and only) output
            # Assumes CPPN final activation (e.g., tanh) is bounded to [-1, 1]
            cppn_output = cppn_output_list[0]

            # 3d. Scale the output from [-1, 1] to the target motor range
            # (cppn_output + 1.0) * 0.5   maps [-1, 1] -> [0, 1]
            scaled_output = self.target_min + \
                (cppn_output + 1.0) * 0.5 * (self.target_max - self.target_min)
            
            target_positions[i] = scaled_output

        # 4. Return the full array of target positions
        # Your `Controller` class will handle smoothing (alpha) and safety clipping.
        return target_positions