import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import gaussian_kde

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

def plot_density_from_c_output(build_dir, binary_name):
    # Get output from C++ binary
    output = build_and_run_c_program(build_dir, binary_name)
    # Parse the output
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)
    y_values = ray_df['y'].values

    # KDE density estimate
    kde = gaussian_kde(y_values)
    y_range = np.linspace(fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'], 500)
    density = kde(y_range)

    plt.figure(figsize=(8, 6))
    plt.plot(density, y_range, label=f'Density of y at x={fiber_info["fiber_length"]}', color='c')
    plt.xlabel('Density')
    plt.ylabel('Y')
    plt.title('Density Function of Y-axis at fiber end')
    plt.grid(True)
    plt.legend()
    plt.show()

# Since you're already in the build directory, use current directory
build_dir = 'build'  # Use current directory instead of 'build'
binary_name = 'raytracing_optical_fiber'
plot_density_from_c_output(build_dir, binary_name)
