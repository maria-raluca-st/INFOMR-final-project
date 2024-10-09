from vedo import Mesh, Box, Sphere, ConvexHull
import numpy as np
from normalize import get_eigenvectors, normalize_shape, get_center_of_mass
"""
Step 3.2: 3D shape descriptors
Compute the following 3D elementary descriptors presented in Module 4: Feature extraction:

surface area
compactness (with respect to a sphere)
3D rectangularity (shape volume divided by OBB volume)
diameter
convexity (shape volume divided by convex hull volume)
eccentricity (ratio of largest to smallest eigenvalues of covariance matrix)
Note that the definitions given in Module 4 are for 2D shapes. You need to adapt them to 3D shapes (easy).

Extract Histograms of small level descriptors per shape, per class (different colors per class, no bins, continous display)

"""


def extract_features(mesh:Mesh):
    cvx = ConvexHull(mesh.vertices)
    diameterRet = get_diameter(mesh,cvx)
    mesh.fill_holes()
    mesh.triangulate()
    ret = {
        "area":mesh.area(),
        "volume":mesh.volume(),
        "rectangularity":get_rectangularity(mesh),
        "compactness":get_compactness(mesh),
        "convexity": get_convexity(mesh,cvx),
        "eccentricity":get_eccentricity(mesh),
        "diameter":diameterRet[0],
        "diameterPts":diameterRet[1],
        "distributions":get_distributions(mesh)
    }
    return ret

def get_surface_area(mesh:Mesh):
    #area = sqrt(fabs(s * (s - a) * (s - b) * (s - c)));
    return mesh.area()

def get_rectangularity(mesh:Mesh):
    #How close is the shape (post normalisation to its oriented bounding box)
    # (shape volume divided by OBB volume)
    bbox = mesh.bounds()
    Dx = np.abs(bbox[0] - bbox[1])  
    Dy = np.abs(bbox[2] - bbox[3])  
    Dz = np.abs(bbox[4] - bbox[5])  
    obbVol = Dx*Dy*Dz
    rectangularity = mesh.volume()/obbVol
    return rectangularity

def get_compactness(mesh:Mesh):
    #How close is the shape to a sphere
    return mesh.area()**3/(36*np.pi*(mesh.volume()**2))

def get_convexity(mesh:Mesh,cvx:ConvexHull):
    #(shape volume divided by convex hull volume)
    convexity = mesh.volume()/cvx.volume()
    return convexity


def get_diameter(mesh:Mesh,cvx:ConvexHull,k=500):
    maxD = 0
    maxP = [None,None]
    if(len(cvx.vertices)<k):
        subs=cvx.vertices
    else:
        subs = cvx.vertices[np.random.choice(cvx.vertices.shape[0], k, replace=False)]
    for v1 in subs:
        for v2 in cvx.vertices:
            d = np.linalg.norm(v1-v2)
            if d>maxD:
                maxD=d
                maxP = [v1,v2]
    return maxD,maxP        
        
        

def get_eccentricity(mesh:Mesh):
    #ratio of largest to smallest eigenvalues of covariance matrix
    _,eigval = get_eigenvectors(mesh)
    mineig = min(eigval)
    maxeig = max(eigval)
    return np.abs(maxeig)/np.abs(mineig)


def get_distributions(mesh:Mesh, show=True):
    com = get_center_of_mass(mesh)
    subsample1= mesh.vertices[np.random.choice(mesh.vertices.shape[0], 5, replace=False), :]
 
    D1 = calc_D1(com,subsample1)

    distributions = {
        "D1":D1
    }
    return distributions


#Calculate distance between the center and a random subset, returns list of euclidian distances
def calc_D1(center, subs):
    ret = []
    for pt in subs:
        ret.append(np.linalg.norm(pt - center))
    return ret
    
    

if __name__ == "__main__":
    boxMesh = normalize_shape(Box(width=1,height=1,length=1).c("Black").wireframe(True))
    sphereMesh = normalize_shape(Sphere(r=1, res=24, quads=False, c='red', alpha=1.0))
    train = normalize_shape(Mesh("..\shapes\Train\D01014.obj"))
    head = normalize_shape(Mesh("..shapes\HumanHead\D00131.obj"))
    insect = normalize_shape(Mesh("..shapes\Insect\D00117.obj"))

    
    print(extract_features(boxMesh))
    print(extract_features(sphereMesh))
    print(extract_features(train))
    print(extract_features(head))
    print(extract_features(insect))
    