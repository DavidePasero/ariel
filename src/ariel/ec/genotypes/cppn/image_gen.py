import numpy as np
from PIL import Image

# --- FIX 1: Update imports to include the new IdManager ---
from ariel.ec.genotypes.cppn.cppn_genome import CPPN_genotype
from ariel.ec.genotypes.cppn.id_manager import IdManager # Import the new class

# --- Image Generation Parameters ---
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
COORD_SCALE = 5.0 # Controls the 'zoom' of the input coordinates
NUM_INPUTS = 3    # X, Y, D (Distance from center)
NUM_OUTPUTS = 3   # R, G, B

# --- FIX 2: Replace global variables with the IdManager instance ---
# The IdManager will now handle all unique ID generation.
id_manager = IdManager()

# --- 1. Initialize the First Genome ---
# The initial genome has a fixed number of nodes and connections.
# We set the ID manager's counters to start *after* these initial genes.
num_initial_nodes = NUM_INPUTS + NUM_OUTPUTS
num_initial_conns = NUM_INPUTS * NUM_OUTPUTS

# Set the manager's state for the very first run.
# In a real evolution, you would load this from a file if it exists.
id_manager._node_id = num_initial_nodes - 1
id_manager._innov_id = num_initial_conns - 1

# Create the initial, randomly weighted CPPN.
# The innovation numbers for the first connections will start at 0.
print("Creating initial random CPPN...")
cppn_genome = Genome.random(
    num_inputs=NUM_INPUTS,
    num_outputs=NUM_OUTPUTS,
    next_node_id=num_initial_nodes, # Not strictly needed if Genome doesn't use it, but good practice
    next_innov_id=0 # Initial innovations start from 0
)

print(f"IdManager initialized. Next node ID: {id_manager._node_id + 1}, Next innovation ID: {id_manager._innov_id + 1}")
# For future mutations, you would now pass the manager's methods:
# e.g., genome.mutate(..., next_node_id_getter=id_manager.get_next_node_id, ...)


def generate_image(genome: CPPN_genotype, width: int, height: int, scale: float = 5.0) -> Image:
    """
    Generates an image by querying the CPPN for the color of each pixel.
    """
    x_coords = np.linspace(-scale, scale, width)
    y_coords = np.linspace(-scale, scale, height)
    image_data = np.zeros((height, width, 3), dtype=np.uint8)

    for y, Y in enumerate(y_coords):
        for x, X in enumerate(x_coords):
            # Calculate the distance from the center (D)
            D = np.sqrt(X**2 + Y**2)
            # Activate the CPPN with the spatial coordinates
            raw_colors = genome.activate([X, Y, D])

            # Map the CPPN's output (typically in [-1, 1]) to an RGB color [0, 255]
            R = np.clip(int(((raw_colors[0] + 1.0) / 2.0) * 255), 0, 255)
            G = np.clip(int(((raw_colors[1] + 1.0) / 2.0) * 255), 0, 255)
            B = np.clip(int(((raw_colors[2] + 1.0) / 2.0) * 255), 0, 255)

            image_data[y, x] = [R, G, B]

    return Image.fromarray(image_data, 'RGB')


# --- 3. Execute Generation ---
print(f"Generating image of size {IMAGE_WIDTH}x{IMAGE_HEIGHT}...")
try:
    generated_image = generate_image(
        cppn_genome,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        scale=COORD_SCALE
    )

    # --- 4. Save the Output ---
    output_filename = "cppn_initial_pattern.png"
    generated_image.save(output_filename)
    print(f"Image successfully generated and saved as {output_filename}")

except Exception as e:
    print(f"An error occurred during image generation: {e}")