import matplotlib.pyplot as plt
import numpy as np
import re

# Bestand openen en cijfers inlezen
with open("plot/data.txt", "r") as f:
    content = f.read()

# Getallen extraheren
y = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", content)))
x = [0] * len(y)

# Interval rond elk punt om lokale dichtheid te bepalen
interval = 0.5

# Bereken voor elk punt hoeveel andere punten binnen dat interval liggen
def compute_density(point, data, interval):
    return sum(1 for p in data if point - interval <= p <= point + interval)

densities = [compute_density(y_i, y, interval) for y_i in y]
norm_densities = np.array(densities) / max(densities)  # Normaliseren voor kleur

# Plot met aangepaste kleurenschaal 
plt.figure(figsize=(2, 6))
scatter = plt.scatter(x, y, c=norm_densities, cmap='Purples', marker='o')

# Opmaak
plt.title("Verticale 1D-plot met dichtheidskleur")
plt.xlabel(" ")
plt.ylabel("Waarde")
plt.xticks([])
plt.grid(True, axis='y')

# Kleurenschaal toevoegen
cbar = plt.colorbar(scatter)
cbar.set_label("Dichtheid")

plt.tight_layout()
plt.show()
