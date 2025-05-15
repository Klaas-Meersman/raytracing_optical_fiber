import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import platform
import matplotlib.cm as cm
import numpy as np


def build_and_run_c_program(build_dir, binary_name):
    # Step 1: Run cmake ..
    print("Running cmake ..")
    subprocess.run(['cmake', '..'], cwd=build_dir, check=True)

    # Step 2: Run build
    print("Running make")
    subprocess.run(['cmake', '--build', '.'], cwd=build_dir, check=True)

    # Step 3: Build correct path
    if platform.system() == "Windows":
        binary_name += ".exe"
        binary_path = os.path.join(build_dir, "Debug", binary_name)
    else:
        binary_path = os.path.join(build_dir, binary_name)

    print(f"Running {binary_path}")
    result = subprocess.run([binary_path], stdout=subprocess.PIPE, text=True, check=False)
    return result.stdout


def read_fiber_and_ray_from_string(csv_string):
    lines = csv_string.strip().splitlines()
    fiber_info = {}
    ray_start_idx = 0

    for i, line in enumerate(lines):
        if line.strip().startswith('id'):
            ray_start_idx = i
            break
        try:
            key, value = line.strip().split(',')
            fiber_info[key] = float(value)
        except ValueError:
            # Als er te veel waarden zijn, is het waarschijnlijk al ray-data
            ray_start_idx = i
            break

    ray_data_str = '\n'.join(lines[ray_start_idx:])
    ray_df = pd.read_csv(io.StringIO(ray_data_str))
    return fiber_info, ray_df



def plot_fiber_and_ray_from_output(output):
    print("----- Raw program output -----")
    print(output)
    print("----- End of program output -----")
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)
    fiber_length = fiber_info['fiber_length']
    fiber_top_y = fiber_info['fiber_top_y']
    fiber_bottom_y = fiber_info['fiber_bottom_y']

    plt.figure(figsize=(12, 7))

    # Plot fiber boundaries
    plt.plot([0, fiber_length], [fiber_top_y, fiber_top_y], 'r-', label='Fiber Top')
    plt.plot([0, fiber_length], [fiber_bottom_y, fiber_bottom_y], 'r-', label='Fiber Bottom')
    plt.axvline(x=fiber_length, color='g', linestyle='--', label='Fiber End')

    # Assign colors
    ray_ids = ray_df['id'].unique()
    num_rays = len(ray_ids)
    colormap = cm.get_cmap('tab20', num_rays)

    for idx, ray_id in enumerate(ray_ids):
        ray_segment = ray_df[ray_df['id'] == ray_id]
        color = colormap(idx)
        plt.plot(ray_segment['x'], ray_segment['y'], '-o', label=f'Ray {ray_id}', color=color)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Optical Fiber Rays')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    build_dir = 'build'
    binary_name = 'raytracing_optical_fiber'
    output = build_and_run_c_program(build_dir, binary_name)
    plot_fiber_and_ray_from_output(output)
