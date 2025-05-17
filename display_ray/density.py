import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def build_and_run_c_program(build_dir, binary_name):
    subprocess.run(['cmake', '..'], cwd=build_dir, check=True)
    subprocess.run(['make'], cwd=build_dir, check=True)
    binary_path = os.path.join(build_dir, binary_name)
    result = subprocess.run([binary_path], stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout

def read_fiber_and_ray_from_string(csv_string):
    lines = csv_string.strip().splitlines()
    fiber_info = {}
    data_start_idx = 0

    # Parse metadata (lines with a single comma)
    for i, line in enumerate(lines):
        parts = line.strip().split(',')
        if len(parts) == 2:
            key, value = parts
            try:
                fiber_info[key.strip()] = float(value.strip())
            except ValueError:
                fiber_info[key.strip()] = value.strip()
        else:
            data_start_idx = i
            break

    # The rest is the data, skip header if present
    ray_data_lines = lines[data_start_idx:]
    if ray_data_lines and all(col in ray_data_lines[0].replace(' ', '').lower() for col in ['x', 'y', 'z']):
        ray_data_lines = ray_data_lines[1:]

    ray_data_str = '\n'.join(ray_data_lines)
    column_names = ['x', 'y', 'z']
    ray_df = pd.read_csv(io.StringIO(ray_data_str), header=None, names=column_names)

    # Convert all columns to float, drop bad rows
    ray_df = ray_df.apply(pd.to_numeric, errors='coerce').dropna()

    return fiber_info, ray_df

def plot_yz_density_at_fiber_end(build_dir, binary_name):
    output = build_and_run_c_program(build_dir, binary_name)
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)

    # Select endpoints (x ≈ fiber_length)
    endpoints_df = ray_df[np.isclose(ray_df['x'], fiber_info['fiber_length'], rtol=1e-3)]
    y = endpoints_df['y'].values
    z = endpoints_df['z'].values

    if len(y) < 10 or len(z) < 10:
        print("⚠️ Too few endpoints, duplicating with jitter for density plot...")
        y = np.tile(y, 50) + np.random.normal(0, 0.02, size=50)
        z = np.tile(z, 50) + np.random.normal(0, 0.02, size=50)

    plt.figure(figsize=(8, 6))
    plt.hist2d(
        y, z,
        bins=300,
        range=[
            [fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y']],
            [fiber_info['fiber_bottom_z'], fiber_info['fiber_top_z']]
        ],
        cmap=LinearSegmentedColormap.from_list(
            "white_to_purple", ["white", "#f2e5ff", "#c084fc", "#7e22ce"]
        ),
        density=False  # Linear scale (default)[1][2][5]
    )
    plt.colorbar(label='Count')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(f"Density plot at fiber end face (x ≈ {fiber_info['fiber_length']})")
    plt.grid(True)
    plt.show()




# Set your build directory and binary name
build_dir = 'build'
binary_name = 'raytracing_optical_fiber'

plot_yz_density_at_fiber_end(build_dir, binary_name)