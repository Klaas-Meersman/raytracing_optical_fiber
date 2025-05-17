import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import re

def build_and_run_c_program(build_dir, binary_name):
    # Run cmake, make, and the binary, capturing its output as a string
    subprocess.run(['cmake', '..'], cwd=build_dir, check=True)
    subprocess.run(['make'], cwd=build_dir, check=True)
    binary_path = os.path.join(build_dir, binary_name)
    result = subprocess.run([binary_path], stdout=subprocess.PIPE, text=True, check=True)
    output = result.stdout

    # Split output into lines
    lines = output.strip().splitlines()
    # Assume the last line is the time string, e.g., "4800 ms"
    last_line = lines[-1]
    # Use regex to extract time in ms
    match = re.match(r"(\d+)\s*ms", last_line)
    if match:
        elapsed_ms = int(match.group(1))
        # Remove the last line from the output for CSV parsing
        output = "\n".join(lines[:-1])
    else:
        elapsed_ms = None  # or handle as needed

    return output, elapsed_ms

def read_fiber_and_ray_from_string(csv_string):
    lines = csv_string.strip().splitlines()
    fiber_info = {}
    ray_start_idx = 0

    # Find the line that starts with 'x' (header for ray data, no id)
    for i, line in enumerate(lines):
        if line.strip().startswith('x'):
            ray_start_idx = i
            break
        parts = line.strip().split(',')
        if len(parts) == 2:
            key, value = parts
            fiber_info[key] = float(value)

    # Parse the ray data as CSV (expects columns: x,y,z)
    ray_data_str = '\n'.join(lines[ray_start_idx:])
    ray_df = pd.read_csv(io.StringIO(ray_data_str))

    return fiber_info, ray_df

def plot_yz_density_at_fiber_end(build_dir, binary_name):
    output, elapsed_ms = build_and_run_c_program(build_dir, binary_name)
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)

    # Select endpoints (x ≈ fiber_length)
    endpoints_df = ray_df[np.isclose(ray_df['x'], fiber_info['fiber_length'], rtol=1e-3)]
    y = endpoints_df['y'].values
    z = endpoints_df['z'].values

    if len(y) < 10 or len(z) < 10:
        print("⚠️ Not enough endpoints, duplicating with jitter for visualization...")
        y = np.tile(y, 50) + np.random.normal(0, 0.02, size=50)
        z = np.tile(z, 50) + np.random.normal(0, 0.02, size=50)

    # 2D histogram for density estimation (no KDE)
    y_bins = np.linspace(fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'], 300)
    z_bins = np.linspace(fiber_info['fiber_bottom_z'], fiber_info['fiber_top_z'], 300)
    density, y_edges, z_edges = np.histogram2d(y, z, bins=[y_bins, z_bins], density=True)

    # Custom colormap: white (low) → purple (high)
    purple_white_cmap = LinearSegmentedColormap.from_list(
        "white_to_purple", ["white", "#f2e5ff", "#c084fc", "#7e22ce"]
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        density.T,
        origin='lower',
        extent=[
            fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'],
            fiber_info['fiber_bottom_z'], fiber_info['fiber_top_z']
        ],
        cmap=purple_white_cmap,
        aspect='auto'
    )
    plt.colorbar(label='Density')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(f"Density plot at fiber end face (x ≈ {fiber_info['fiber_length']})")
    plt.grid(True)
    plt.show()

    if elapsed_ms is not None:
        print(f"Algorithm time: {elapsed_ms} ms")
    else:
        print("Algorithm time: Not found")

# Set your build directory and binary name
build_dir = 'build'
binary_name = 'raytracing_optical_fiber'

plot_yz_density_at_fiber_end(build_dir, binary_name)
