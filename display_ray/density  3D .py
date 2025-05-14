import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import gaussian_kde
import platform
from matplotlib.colors import LinearSegmentedColormap
purple_cmap = LinearSegmentedColormap.from_list("white_to_purple", ["white", "purple"])
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D



def build_and_run_c_program(build_dir, binary_name):
    if platform.system() == "Windows":
        binary_name += '.exe'
        binary_path = os.path.join(build_dir, "Debug", binary_name)
    else:
        binary_path = os.path.join(build_dir, binary_name)

    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Binary not found at: {binary_path}")
    
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


def plot_yz_density_at_fiber_end(build_dir, binary_name):
    output = build_and_run_c_program(build_dir, binary_name)
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)

    # Filter eindpunten (x ≈ fiber_length)
    eindpunten_df = ray_df[np.isclose(ray_df['x'], fiber_info['fiber_length'], rtol=1e-3)]
    y = eindpunten_df['y'].values
    z = eindpunten_df['z'].values

    # Fallback als er te weinig data is
    if len(y) < 2 or len(z) < 2:
        print("⚠️ Te weinig eindpunten, dupliceren met jitter voor KDE...")
        y = np.tile(y, 10) + np.random.normal(0, 0.01, size=10)
        z = np.tile(z, 10) + np.random.normal(0, 0.01, size=10)

    # 2D KDE
    values = np.vstack([y, z])
    kde = gaussian_kde(values)
    
    y_grid = np.linspace(fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'], 300)
    z_grid = np.linspace(fiber_info['fiber_left_z'], fiber_info['fiber_right_z'], 300)
    Y, Z = np.meshgrid(y_grid, z_grid)
    grid_coords = np.vstack([Y.ravel(), Z.ravel()])
    density = kde(grid_coords).reshape(Y.shape)

    # Plotten
    plt.figure(figsize=(8, 6))
    plt.imshow(
        density.T, 
        origin='lower', 
        extent=[fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'], fiber_info['fiber_left_z'], fiber_info['fiber_right_z']],
        cmap=purple_cmap, 
        aspect='auto'
    )
    plt.colorbar(label='Dichtheid')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(f"Dichtheidsplot van stralen op eindvlak (x ≈ {fiber_info['fiber_length']})")
    plt.grid(True)
    plt.show()


def plot_3d_ray_path_in_fiber_plotly(build_dir, binary_name):
    output = build_and_run_c_program(build_dir, binary_name)
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)

    # Fiber dimensies en ranges
    fiber_length = fiber_info['fiber_length']
    y_min, y_max = fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y']
    z_min, z_max = fiber_info['fiber_left_z'], fiber_info['fiber_right_z']

    x_range = fiber_length
    y_range = y_max - y_min
    z_range = z_max - z_min

    # 3D scatter
    fig = go.Figure(go.Scatter3d(
        x=ray_df['x'], y=ray_df['y'], z=ray_df['z'],
        mode='markers+lines', marker=dict(size=3, color='red'),
        line=dict(width=1), name='Stralenpad'
    ))

    # Fiber box (wireframe)
    for x_fixed in (0, fiber_length):
        fig.add_trace(go.Mesh3d(
            x=[x_fixed]*4,
            y=[y_min, y_max, y_max, y_min],
            z=[z_min, z_min, z_max, z_max],
            opacity=0.1, color='cyan', name='Fibervlak'
        ))

    # Stel de scene_aspectratio in op de werkelijke verhoudingen
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, fiber_length], title='X'),
            yaxis=dict(range=[y_min, y_max], title='Y'),
            zaxis=dict(range=[z_min, z_max], title='Z'),
            aspectmode='manual',
            aspectratio=dict(
                x=abs(x_range),  # Zorg ervoor dat de waarde positief is
                y=abs(y_range),  # Zorg ervoor dat de waarde positief is
                z=abs(z_range)   # Zorg ervoor dat de waarde positief is
            )
        ),
    title='3D reflectiepunten in fiber (Plotly)'
    )
    fig.show()
# Pas aan naargelang je build directory
build_dir = 'build'
binary_name = 'raytracing_optical_fiber'

#plot_yz_density_at_fiber_end(build_dir, binary_name)
plot_3d_ray_path_in_fiber_plotly(build_dir, binary_name)
