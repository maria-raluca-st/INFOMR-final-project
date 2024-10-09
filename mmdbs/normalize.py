from vedo import Mesh, LinearTransform
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



def get_center_of_mass(mesh:Mesh):
    #Naive COmputation: np.mean(mesh.vertices,axis=0)
    return mesh.center_of_mass()
    #Bary Center is the mean of the triangle centers weighed by their area
    meshVolume = 0
    temp = (0, 0, 0)

    for i1, i2, i3 in mesh.cells:
        v1, v2, v3 = mesh.vertices[i1], mesh.vertices[i2], mesh.vertices[i3]
        center = (v1 + v2 + v3) / 4  # center of tetrahedron
        volume = np.dot(v1, np.cross(v2, v3)) / 6  # signed volume of tetrahedron
        meshVolume += volume
        temp += center * volume

    meshCenter = temp / meshVolume
    return meshCenter


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
    # print("Com Pre", com)
    wMesh.vertices -= com
    # LT = LinearTransform()
    # LT.translate(wMesh.transform.position-com)
    # LT.move(wMesh)
    # com2 = get_center_of_mass(wMesh)
    # print("Com Post", com2)
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
    eigenvectors, eigenvalues = get_eigenvectors(wMesh)
    ranking = np.argpartition(eigenvalues, 2)
    com = get_center_of_mass(mesh)
    aligned_matrix = [
        eigenvectors[ranking[2]],
        eigenvectors[ranking[1]],
        np.cross(eigenvectors[ranking[2]], eigenvectors[ranking[1]]),
    ]
    wMesh.vertices = np.dot(
        wMesh.vertices - com, np.transpose(aligned_matrix)
    )  # Simple Crossproduct
    return wMesh
    nverts = []  # Manual Method
    for v in wMesh.vertices:
        xpos = np.dot(v, eigenvectors[ranking[2]])
        ypos = np.dot(v, eigenvectors[ranking[1]])
        zpos = v - np.cross(v, eigenvectors[ranking[0]])
        nverts.append([xpos, ypos, zpos])
    wMesh.vertices = nverts
    return wMesh


def normalize_vertices(mesh: Mesh, target_range=(5000, 8000), max_fraction=0.7, max_iters=10):
    """
    Redistributes vertices so that they are within a target range and more uniformly distributed across the object.
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
    
        v1, v2, v3 = mesh.vertices[i1], mesh.vertices[i2], mesh.vertices[i3] # vertices
        Ct = (v1 + v2 + v3) / 3.0 # center of triangle
      
        for i in range(3): # loop over x, y, z
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
    """

    bbox_min, bbox_max = mesh.bounds()[:3], mesh.bounds()[3:]  # bounding box of mesh
    
    Dx = np.abs(bbox_max[0] - bbox_min[0])  
    Dy = np.abs(bbox_max[1] - bbox_min[1])  
    Dz = np.abs(bbox_max[2] - bbox_min[2])  
    
    Dmax = max(Dx, Dy, Dz) #   # largest dimension of bounding box
    
    s = 1 / Dmax # scaling factor
    
    mesh.scale(s) # resizing mesh
    
    return mesh

    Dx = bbox_max[0] - bbox_min[0]
    Dy = bbox_max[1] - bbox_min[1]
    Dz = bbox_max[2] - bbox_min[2]

    Dmax = max(Dx, Dy, Dz)  #   # largest dimension of bounding box

    s = 1 / Dmax  # scaling factor

    mesh.scale(s)  # resizing mesh

    return mesh