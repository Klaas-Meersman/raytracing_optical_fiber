import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

def build_and_run_c_program(build_dir, binary_name):
    # Run cmake, make, and the binary, capturing its output as a string
    subprocess.run(['cmake', '..'], cwd=build_dir, check=True)
    subprocess.run(['make'], cwd=build_dir, check=True)
    binary_path = os.path.join(build_dir, binary_name)
    result = subprocess.run([binary_path], stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout

def read_fiber_and_ray_from_string(csv_string):
    lines = csv_string.strip().splitlines()
    fiber_info = {}
    ray_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('x'):
            ray_start_idx = i
            break
        key, value = line.strip().split(',')
        fiber_info[key] = float(value)
    ray_data_str = '\n'.join(lines[ray_start_idx:])
    ray_df = pd.read_csv(io.StringIO(ray_data_str))
    return fiber_info, ray_df

def plot_2d_density_from_c_output(build_dir, binary_name):
    # Get output from C++ binary
    output = build_and_run_c_program(build_dir, binary_name)
    # Parse the output
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)
    y_values = ray_df['y'].values
    z_values = ray_df['z'].values

    # Define the range for y and z based on fiber info
    y_min, y_max = fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y']
    z_min, z_max = fiber_info['fiber_bottom_z'], fiber_info['fiber_top_z']

    # Plot a high-resolution 2D histogram (heatmap) of the ray endpoints in grayscale
    plt.figure(figsize=(8, 6))
    plt.hist2d(
        y_values, z_values,
        bins=500,  # Higher resolution
        range=[[y_min, y_max], [z_min, z_max]],
        cmap='Greys'  # Use grayscale
    )
    plt.colorbar(label='Number of rays')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('Density of Ray Endpoints at Fiber End (Y-Z plane)')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Usage
build_dir = 'build'
binary_name = 'raytracing_optical_fiber'
plot_2d_density_from_c_output(build_dir, binary_name)
