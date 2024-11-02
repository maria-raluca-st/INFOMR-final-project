from vedo import Plotter, Mesh, dataurl


file_low = "../shapes/Door/D01062.obj"
file_high = "../shapes/Apartment/D00045.obj"
file_non_uniform = "../shapes/Tool/m1108.obj"

# Load different types of meshes
low_cell_mesh = Mesh(file_low)  # A model with relatively fewer cells
high_cell_mesh = Mesh(file_high)  # A model with more cells
non_uniform_mesh = Mesh(file_non_uniform)  # A mesh with a more non-uniform distribution

# Analyze the cell count and distribution
print(f"Low Cell Mesh: {low_cell_mesh.ncells} cells")
print(f"High Cell Mesh: {high_cell_mesh.ncells} cells")
print(f"Non-Uniform Mesh: {non_uniform_mesh.ncells} cells")

# Optionally visualize the meshes with cell coloring to see the structure
plt = Plotter(N=3, axes=11)

plt.show([low_cell_mesh.c("lightgray"), "Low Cell Count Mesh"], at=0)
plt.show([high_cell_mesh.c("lightgray"), "High Cell Count Mesh"], at=1)
plt.show([non_uniform_mesh.c("lightgray"), "Non-Uniform Mesh"], at=2)

plt.interactive().close()

# Low Cell Mesh: 252 cells
# High Cell Mesh: 21021 cells
# Non-Uniform Mesh: 3756 cells
