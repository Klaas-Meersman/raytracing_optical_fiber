import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import gaussian_kde
import platform
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap




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

    # Zoek naar de regel die begint met 'id' (de header van de ray-data)
    for i, line in enumerate(lines):
        if line.strip().startswith('id'):
            ray_start_idx = i
            break
        parts = line.strip().split(',')
        if len(parts) == 2:
            key, value = parts
            fiber_info[key] = float(value)

    # Verwerk de ray-data als een CSV
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

    if len(y) < 10 or len(z) < 10:
        print("⚠️ Te weinig eindpunten, dupliceren met jitter voor KDE...")
        y = np.tile(y, 50) + np.random.normal(0, 0.02, size=50)
        z = np.tile(z, 50) + np.random.normal(0, 0.02, size=50)

    # 2D KDE met strakkere bandwidth
    values = np.vstack([y, z])
    kde = gaussian_kde(values, bw_method=0.2)

    y_grid = np.linspace(fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'], 300)
    z_grid = np.linspace(fiber_info['fiber_right_z'], fiber_info['fiber_left_z'], 300)  # Z-as normaal (onder naar boven)
    Y, Z = np.meshgrid(y_grid, z_grid)
    grid_coords = np.vstack([Y.ravel(), Z.ravel()])
    density = kde(grid_coords).reshape(Y.shape)

    # 🔵 Custom colormap: wit (laag) → paars (hoog)
    purple_white_cmap = LinearSegmentedColormap.from_list("white_to_purple", ["white", "#f2e5ff", "#c084fc", "#7e22ce"])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(
        density.T,
        origin='lower',  # juiste oriëntatie, Z-as van onder naar boven
        extent=[fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y'], 
                fiber_info['fiber_left_z'], fiber_info['fiber_right_z']],
        cmap=purple_white_cmap,
        aspect='auto'
    )
    plt.colorbar(label='Dichtheid')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(f"Dichtheidsplot op fiber-eindvlak (x ≈ {fiber_info['fiber_length']})")
    plt.grid(True)
    plt.show()


def plot_3d_ray_path_in_fiber_plotly_with_edges(build_dir, binary_name):
    output = build_and_run_c_program(build_dir, binary_name)
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)

    # Fiber dimensies
    fiber_length = fiber_info['fiber_length']
    y_min, y_max = fiber_info['fiber_bottom_y'], fiber_info['fiber_top_y']
    z_min, z_max = fiber_info['fiber_right_z'], fiber_info['fiber_left_z']

    fig = go.Figure()

    # Groepeer op 'id' en plot iedere ray met een unieke kleur
    ray_ids = ray_df['id'].unique()
    colors = px.colors.qualitative.Plotly  # Gebruik standaard plotly kleuren

    for i, ray_id in enumerate(ray_ids):
        ray_data = ray_df[ray_df['id'] == ray_id]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=ray_data['x'], y=ray_data['y'], z=ray_data['z'],
            mode='lines+markers',
            marker=dict(size=3),
            line=dict(color=color, width=2),
            name=f'Ray {ray_id}'
        ))

    # Fiber kubusranden tekenen
    corners = np.array([
        [0, y_min, z_min], [0, y_min, z_max], [0, y_max, z_min], [0, y_max, z_max],
        [fiber_length, y_min, z_min], [fiber_length, y_min, z_max], 
        [fiber_length, y_max, z_min], [fiber_length, y_max, z_max]
    ])

    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],       # voorzijde
        [4, 5], [5, 7], [7, 6], [6, 4],       # achterzijde
        [0, 4], [1, 5], [2, 6], [3, 7]        # verbindingen
    ]

    for edge in edges:
        x_vals = [corners[edge[0]][0], corners[edge[1]][0]]
        y_vals = [corners[edge[0]][1], corners[edge[1]][1]]
        z_vals = [corners[edge[0]][2], corners[edge[1]][2]]
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, fiber_length], title='X'),
            yaxis=dict(range=[y_min, y_max], title='Y'),
            zaxis=dict(range=[z_min, z_max], title='Z'),
            aspectmode='manual',
            aspectratio=dict(
                x=abs(fiber_length), 
                y=abs(y_max - y_min), 
                z=abs(z_max - z_min)
            )
        ),
        title='3D ray tracing in fiber per ray ID',
        legend=dict(itemsizing='constant')
    )

    fig.show()



# Pas aan naargelang je build directory
build_dir = 'build'
binary_name = 'raytracing_optical_fiber'


plot_3d_ray_path_in_fiber_plotly_with_edges(build_dir, binary_name)
plot_yz_density_at_fiber_end(build_dir, binary_name)
