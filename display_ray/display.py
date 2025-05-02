import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import pandas as pd
import os


# Read coordinates from CSV file
# Assuming the CSV has two columns named 'x' and 'y'
def plot_ray_from_csv(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Extract x and y values
    x_vals = df['x'].values
    y_vals = df['y'].values
    
    # Plotting the ray
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
    plt.title('Plot of the ray coordinates from CSV')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# Usage
print("Current working directory:", os.getcwd())
plot_ray_from_csv('/home/klaas/github/raytracing_optical_fiber/display_ray/rays.csv')
