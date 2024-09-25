from vedo import Mesh, LinearTransform
import numpy as np

def normalize_shape(mesh:Mesh):
    """
    Applies several steps of normalisation to provide a comparable baseline for each mesh
    ----------------------------
    Args:
        Vedo Mesh
    Returns:
        Vedo Mesh
    """
    npos = normalize_position(mesh)
    nrot = normalize_pose(npos)
    nver = normalize_vertices(nrot)
    nflip = normalize_flip(nver)
    nsize = normalize_scale(nflip)
    return nsize

def get_eigenvectors(mesh:Mesh):
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
    return eigenvectors, eigenvalues

    

def normalize_position(mesh:Mesh,inplace=True):
    """
    Shifts input mesh so that its center of mass coincides with the origin
    ----------------------------
    Args:
        Vedo Mesh
    Returns:
        Vedo Mesh
    """
    if(not inplace):
        wMesh=mesh.copy()
    else:
        wMesh=mesh
    LT = LinearTransform()
    LT.translate(wMesh.transform.position-wMesh.center_of_mass())
    LT.move(wMesh)
    return wMesh

def normalize_pose(mesh:Mesh,inplace=True):
    """
    Rotates mesh so that its major axes are aligned with the unit axes. 
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh
    """
    if(not inplace):
        wMesh=mesh.copy()
    else:
        wMesh=mesh
    eigenvectors,eigenvalues = get_eigenvectors(wMesh)
    ranking = np.argpartition(eigenvalues, 2)
    aligned_matrix = [eigenvectors[ranking[2]],
                      eigenvectors[ranking[1]],
                      eigenvectors[ranking[0]]]
    wMesh.vertices = np.dot(wMesh.vertices,np.transpose(aligned_matrix))

def normalize_vertices(mesh:Mesh):
    """
    Redistributes vertices so that they are within a target range and more uniformly distributed across the object. 
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh with vertices redistributed
    """
    return mesh

def normalize_flip(mesh:Mesh):
    """
    Rotates mesh so that "heaviest" side is within one quadrant, ensuring similar orientation
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh flipped
    """
    return mesh

def normalize_scale(mesh:Mesh):
    """
    Resizes mesh so that its largest axis has a size of one 
    ----------------------------
    Input:
        Vedo Mesh
    Returns:
        Vedo Mesh with size normalized
    """
    return mesh
