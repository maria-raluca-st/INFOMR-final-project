from vedo import Mesh, LinearTransform

import argparse


def normalize_vertices(mesh: Mesh, target_range=(5000, 8000), max_reduction=0.7, max_iters=10):
    """
    Redistributes vertices so that they are within a target range and more uniformly distributed across the object.
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh with vertices redistributed
    """
    _min, _max = target_range
    i = 0
    while not (_max >= mesh.nvertices >= _min) and (i := i+1) < max_iters:
        # print("Dead")
        if mesh.nvertices > _max:
            reduction_factor = _max / mesh.nvertices
            mesh.decimate(
                fraction=max(max_reduction, reduction_factor)
            )  # Slow reduction should be better
        else:
            mesh.subdivide(method=2)  # Slow/adaptive should be better

def main():
    parser = argparse.ArgumentParser(description="Script for normalising vector counts")
    parser.add_argument("path", type=str)
    mesh_path = parser.parse_args().path.replace("\\", "/")
    mesh = Mesh(mesh_path)
    print(mesh.nvertices)
    normalize_vertices(mesh)
    print(mesh.nvertices)

if __name__ == '__main__':
    main()
