from vedo import Mesh, LinearTransform
import pymeshlab
import numpy as np


def normalize_shape(mesh: Mesh, inplace=True):
    """
    Applies several steps of normalisation to provide a comparable baseline for each mesh
    ----------------------------
    Args:
        Vedo Mesh
    Returns:
        Vedo Mesh
    """
    if not inplace:
        wMesh = mesh.copy()
    else:
        wMesh = mesh
    # print("Step vertices"
    normalize_vertices(wMesh)
    normalize_position(wMesh)
    normalize_pose(wMesh)
    normalize_flip(wMesh)
    normalize_scale(wMesh)
    return wMesh


def get_eigenvectors(mesh: Mesh):
    """
    Returns eigenvectors and eigenvalues of a mesh
    ----------------------------
    Args:
        Vedo Mesh
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    A_cov = np.cov(np.transpose(mesh.vertices))  # 3x3 matrix
    # computes the eigenvalues and eigenvectors for the
    # covariance matrix. See documentation at
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)
    # print(eigenvectors,eigenvalues) TODO: remove?
    return eigenvectors, eigenvalues


def get_center_of_mass(mesh: Mesh):
    # Naive Computation: np.mean(mesh.vertices,axis=0)
    return mesh.center_of_mass()


def normalize_position(mesh: Mesh, inplace=True):
    """
    Shifts input mesh so that its center of mass coincides with the origin
    ----------------------------
    Args:
        Vedo Mesh
    Returns:
        Vedo Mesh
    """
    if not inplace:
        wMesh = mesh.copy()
    else:
        wMesh = mesh
    com = get_center_of_mass(wMesh)
    wMesh.vertices -= com
    return wMesh


def normalize_pose(mesh: Mesh, inplace=True):
    """
    Rotates mesh so that its major axes are aligned with the unit axes.
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh
    """
    if not inplace:
        wMesh = mesh.copy()
    else:
        wMesh = mesh
    lab_mesh = pymeshlab.MeshSet()
    lab_mesh.add_mesh(pymeshlab.Mesh(wMesh.vertices))
    lab_mesh.apply_filter(
        "compute_matrix_by_principal_axis", pointsflag=True, freeze=True, alllayers=True
    )
    wMesh.vertices = lab_mesh.current_mesh().vertex_matrix()
    return wMesh


def normalize_vertices(mesh: Mesh, target_range=(5000, 8000), max_fraction=0.7, max_iters=50):
    """
    Redistributes vertices so that they are within a target range and more uniformly distributed across the object.
    Also fills holes and normalises polygonal orientation.
    ----------------------------
    Input:
        Vedo Mesh
        target_range: tuple with inclusive minimum and maximum number of verts
        max_fraction: Maximum fraction of vertices leftover after decimation. Should ensure slower, gradual decimation.
        max_iters: Max number of decimation/subdivision steps to avoid unlimited looping.
    Returns:
        Vedo Mesh with vertices redistributed
    """
    _min, _max = target_range
    i = 0
    while not (_max >= mesh.nvertices >= _min) and (i := i+1) < max_iters:
        if mesh.nvertices > _max:
            reduction_factor = _max / mesh.nvertices
            mesh.decimate(
                fraction=max(max_fraction, reduction_factor)
            )  # Slow reduction should be better
        else:
            mesh.subdivide(method=2)  # Slow/adaptive should be better
    mesh.fill_holes()
    mesh.compute_normals()  # Normalises polygonal orientations


def normalize_flip(mesh: Mesh):
    """
    Flips the mesh so that the majority of the shape's mass is in the positive half of each axis (x, y, z).
    If the mass is mostly in the negative half, mirror the mesh along that axis.
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh flipped along necessary axes
    """

    # Init flipping test values for x, y, z
    f = [0.0, 0.0, 0.0]

    for i1, i2, i3 in mesh.cells:  # loop over list of triangles (indices of vertices)

        v1, v2, v3 = mesh.vertices[i1], mesh.vertices[i2], mesh.vertices[i3]  # vertices
        Ct = (v1 + v2 + v3) / 3.0  # center of triangle

        for i in range(3):  # loop over x, y, z
            f[i] += np.sign(Ct[i]) * (Ct[i] ** 2)

    flip_signs = [np.sign(f[0]), np.sign(f[1]), np.sign(f[2])]
    scale_factors = [flip_signs[0], flip_signs[1], flip_signs[2]]

    mesh.scale(scale_factors)  # flip mesh if necessary

    return mesh


def normalize_scale(mesh: Mesh):
    """
    Resizes mesh so that its largest axis has a size of one.
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh with size normalized

        [xmin,xmax, ymin,ymax, zmin,zmax]
    """

    bbox = mesh.bounds()

    Dx = np.abs(bbox[0] - bbox[1])
    Dy = np.abs(bbox[2] - bbox[3])
    Dz = np.abs(bbox[4] - bbox[5])

    Dmax = max(Dx, Dy, Dz)  #   # largest dimension of bounding box

    s = 1 / Dmax  # scaling factor

    mesh.scale(s)  # resizing mesh

    return mesh
