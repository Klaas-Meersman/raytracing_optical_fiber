import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import gaussian_kde
import platform

def build_and_run_c_program(build_dir, binary_name):
    # Zorg dat je Windows .exe pad gebruikt
    if platform.system() == "Windows":
        binary_name += '.exe'
        binary_path = os.path.join(build_dir, "Debug", binary_name)
    else:
        binary_path = os.path.join(build_dir, binary_name)

    # Check of de binary bestaat
    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Binary not found at: {binary_path}\nHeb je het project gebouwd met CMake?")

    print(f"Running binary: {binary_path}")
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


# Stel build map in waar .exe gegenereerd werd
build_dir = 'build'
binary_name = 'raytracing_optical_fiber'

plot_density_from_c_output(build_dir, binary_name)
