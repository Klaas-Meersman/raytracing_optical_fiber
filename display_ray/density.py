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

    print("📤 Output van C++ programma:\n")
    print(output)

    # Parse the output
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)

    # Alleen y-waarden waar x gelijk is aan de fiber_length
    eind_y_df = ray_df[np.isclose(ray_df['x'], fiber_info['fiber_length'], rtol=1e-3)]
    y_values = eind_y_df['y'].values
    if len(y_values) < 2:
        print("⚠️ Te weinig eindpunten, dupliceren met kleine jitter voor KDE...")
        y_values = np.tile(y_values, 10)  # 10 keer kopiëren
        y_values = y_values + np.random.normal(0, 0.02, size=y_values.shape)
    # Debug prints
    print("🧵 Fiber informatie:")
    for k, v in fiber_info.items():
        print(f"  {k}: {v}")

    print("\n🔍 Laatste ray-punten (x ≈ fiber_length):")
    print(eind_y_df)

    print(f"\n📊 Statistieken over y-waarden aan einde:")
    print(f"  Aantal stralen: {len(y_values)}")
    print(f"  Min y: {np.min(y_values)}")
    print(f"  Max y: {np.max(y_values)}")
    print(f"  Gemiddelde y: {np.mean(y_values)}")
    print(f"  Mediaan y: {np.median(y_values)}")

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
