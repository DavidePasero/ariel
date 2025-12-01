import mujoco as mj
import mujoco.viewer
import numpy as np
from pathlib import Path
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

# --- 1. Imports from your provided file ---
# (Using your exact import paths)
from src.ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.environments import SimpleFlatWorld
from ariel.ec.genotypes.cppn.cppn_genome import CPPN_genotype

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)

# =========================================================================
# 2. CPPN CONTROLLER CALLBACK CLASS
# (This class is the adapter between your Genome and Controller)
# =========================================================================
class CppnControllerCallback:
    """
    A callable object that implements the controller_callback_function
    by querying a 'Genome' (CPPN) instance.
    """
    def __init__(self,
                 motor_coordinates: np.ndarray,
                 time_frequency: float = 0.01,
                 bias_input: float = 1.0):
        self.motor_coords = motor_coordinates
        self.num_actuators = motor_coordinates.shape[0]
        self.freq = time_frequency
        self.bias = bias_input
        self.genome: CPPN_genotype | None = None
        self.target_min = -np.pi / 2
        self.target_max = np.pi / 2
        
        # We expect 6 inputs: x, y, z, t_sin, t_cos, bias
        self.expected_inputs = 6 
        print(f"CppnCallback: Initialized for {self.num_actuators} motors.")

    def set_genome(self, genome: CPPN_genotype) -> None:
        """Sets the active Genome (CPPN) for the next simulation."""
        self.genome = genome
        print("CppnCallback: New genome has been set.")

    def __call__(self, model: mj.MjModel, data: mj.MjData, *args, **kwargs) -> np.ndarray:
        """This is called by the `Controller` class."""
        if self.genome is None:
            raise RuntimeError("CPPN callback called before 'set_genome()'.")

        t = data.time
        t_sine = np.sin(self.freq * t)
        t_cosine = np.cos(self.freq * t)
        target_positions = np.zeros(self.num_actuators)

        for i in range(self.num_actuators):
            coords = self.motor_coords[i]
            cppn_inputs = [
                coords[0], coords[1], coords[2],
                t_sine, t_cosine, self.bias
            ]
            
            # Activate network (expects 1 output)
            cppn_output = self.genome.activate(cppn_inputs)[0]
            
            # Scale output from [-1, 1] to motor range
            scaled_output = self.target_min + \
                (cppn_output + 1.0) * 0.5 * (self.target_max - self.target_min)
            target_positions[i] = scaled_output

        return target_positions

# =========================================================================
# 3. MAIN SIMULATION SCRIPT
# =========================================================================
def main():
    # --- 1-4. Initialise World, Robot, and MuJoCo ---
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, position=[0, 0, 0.0])
    model = world.spec.compile()
    data = mj.MjData(model)
    print(f"Model compiled with {model.nu} actuators.")
    assert model.nu == 8, f"Gecko robot should have 8 actuators, but model has {model.nu}"
    
    # --- 5. Initialise data tracking ---
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )
        # Non-default VideoRecorder options
    video_recorder = VideoRecorder(output_folder=DATA)


    # --- 7. Create a Genome ---
    # (Do this before the controller setup)
    N_INPUTS = 6
    N_OUTPUTS = 1
    initial_node_id_count = N_INPUTS + N_OUTPUTS
    initial_innov_id_count = N_INPUTS * N_OUTPUTS
    my_genome = CPPN_genotype.random(
        num_inputs=N_INPUTS, 
        num_outputs=N_OUTPUTS, 
        next_node_id=initial_node_id_count,
        next_innov_id=initial_innov_id_count
    )

    # --- 9a. Set up Tracker and Reset Data ---
    # (This MUST happen BEFORE coordinate extraction)
    tracker.setup(world.spec, data)  # Use my_controller.tracker.setup if tracker is inside
    mujoco.mj_resetData(model, data)
    
    # --- 6. (MOVED & FIXED) AUTOMATICALLY EXTRACT MOTOR COORDINATES ---
    print("Extracting motor coordinates from t=0 data...")
    motor_coordinates = []

    # Get the global position of the 'core' geom's body at t=0
    core_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "robot1_core")
    if core_geom_id == -1:
        raise ValueError("Could not find a geom named 'core' to use as the robot's origin.")
    
    core_body_id = model.geom_bodyid[core_geom_id]
    core_pos_t0 = data.xpos[core_body_id].copy() # Global position of core at t=0

    for i in range(model.nu):
        # Ensure actuator is a joint actuator
        if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_JOINT:
            
            # Get the joint ID from the actuator's transmission
            joint_id = model.actuator_trnid[i, 0]
            
            # Get the joint's global anchor position from DATA (not model)
            joint_pos_t0 = data.xanchor[joint_id].copy()
            
            # Calculate the joint's position relative to the core
            motor_pos_relative = joint_pos_t0 - core_pos_t0
            motor_coordinates.append(motor_pos_relative)
            
            joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            print(f"  > Motor {i} (joint '{joint_name}'): rel_pos {motor_pos_relative}")
        
        else:
            print(f"  > Warning: Actuator {i} is not a joint actuator. Skipping.")

    motor_coordinates = np.array(motor_coordinates)
    assert len(motor_coordinates) == 8, \
        f"Expected 8 joint actuators, but found {len(motor_coordinates)}"


    # --- 8. Instantiate the Callback and Controller ---
    cppn_callback = CppnControllerCallback(
        motor_coordinates=motor_coordinates, # Pass the *correct* coordinates
        time_frequency=0.01
    )
    cppn_callback.set_genome(my_genome)

    my_controller = Controller(
        controller_callback_function=cppn_callback,
        tracker=tracker,
    )

    # --- 9b. Set MuJoCo Callback ---
    # (The reset is already done, so we just set the control)
    mujoco.set_mjcb_control(my_controller.set_control)
    video_renderer(
        model,
        data,
        duration=10,
        video_recorder=video_recorder,
    )

    # --- 10. Run the Simulation ---
    # duration = 1000  # (seconds)
    # print("\nStarting simulation... (Close viewer window to exit)")
    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     while viewer.is_running() and data.time < duration:
    #         mujoco.mj_step(model, data)
    #         viewer.sync()

    # --- 11. Cleanup ---
    mujoco.set_mjcb_control(None)
    print("Simulation complete.")


if __name__ == "__main__":
    main()