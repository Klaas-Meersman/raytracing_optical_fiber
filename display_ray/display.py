import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io

def build_and_run_c_program(build_dir, binary_name):
    # Step 1: Run cmake ..
    print("Running cmake ..")
    subprocess.run(['cmake', '..'], cwd=build_dir, check=True)
    # Step 2: Run make
    print("Running make")
    subprocess.run(['make'], cwd=build_dir, check=True)
    # Step 3: Run the binary and capture output
    binary_path = os.path.join(build_dir, binary_name)
    print(f"Running {binary_path}")
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

def plot_fiber_and_ray_from_output(output):
    fiber_info, ray_df = read_fiber_and_ray_from_string(output)
    fiber_length = fiber_info['fiber_length']
    fiber_top_y = fiber_info['fiber_top_y']
    fiber_bottom_y = fiber_info['fiber_bottom_y']

    plt.figure(figsize=(10, 6))
    plt.plot([0, fiber_length], [fiber_top_y, fiber_top_y], 'r-', label='Fiber Top')
    plt.plot([0, fiber_length], [fiber_bottom_y, fiber_bottom_y], 'r-', label='Fiber Bottom')
    plt.axvline(x=fiber_length, color='g', linestyle='--', label='Fiber End')
    plt.plot(ray_df['x'], ray_df['y'], 'b-o', label='Ray')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    build_dir = 'build'
    binary_name = 'raytracing_optical_fiber'
    output = build_and_run_c_program(build_dir, binary_name)
    plot_fiber_and_ray_from_output(output)
