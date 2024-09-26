from vedo import Mesh, LinearTransform
import numpy as np

def normalize_shape(mesh:Mesh,inplace=True):
    """
    Applies several steps of normalisation to provide a comparable baseline for each mesh
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
    normalize_position(wMesh)
    normalize_pose(wMesh)
    normalize_vertices(wMesh)
    normalize_flip(wMesh)
    normalize_scale(wMesh)
    return wMesh

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



def get_center_of_mass(mesh:Mesh):
    #Naive COmputation: np.mean(mesh.vertices,axis=0)
    #return mesh.center_of_mass()
    #Bary Center is the mean of the triangle centers weighed by their area
    meshVolume = 0
    temp = (0,0,0)

    for i1,i2,i3 in mesh.cells:
        v1,v2,v3 = mesh.vertices[i1],mesh.vertices[i2],mesh.vertices[i3]
        center = (v1 + v2 + v3) / 4          #center of tetrahedron
        volume = np.dot(v1, np.cross(v2, v3)) / 6  #signed volume of tetrahedron
        meshVolume += volume
        temp += center * volume

    meshCenter = temp / meshVolume
    return meshCenter

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
    com = get_center_of_mass(wMesh)
    #print("Com Pre", com)
    wMesh.vertices-=com
    #LT = LinearTransform()
    #LT.translate(wMesh.transform.position-com)
    #LT.move(wMesh)
    #com2 = get_center_of_mass(wMesh)
    #print("Com Post", com2)
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
    com = get_center_of_mass(mesh)
    aligned_matrix = [eigenvectors[ranking[2]],
                      eigenvectors[ranking[1]],
                      np.cross(eigenvectors[ranking[2]],eigenvectors[ranking[1]])]
    wMesh.vertices = np.dot(wMesh.vertices-com,np.transpose(aligned_matrix)) #Simple Crossproduct
    return wMesh
    nverts = [] #Manual Method
    for v in wMesh.vertices:
        xpos = np.dot(v,eigenvectors[ranking[2]])
        ypos = np.dot(v,eigenvectors[ranking[1]])
        zpos = v - np.cross(v,eigenvectors[ranking[0]])
        nverts.append([xpos,ypos,zpos])
    wMesh.vertices = nverts
    return wMesh
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
