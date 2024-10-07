from vedo import Mesh, Box, Sphere, ConvexHull
import numpy as np

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
    mesh.triangulate()
    ret = {
        "area":mesh.area(),
        "volume":mesh.volume(),
        "rectangularity":get_rectangularity(mesh),
        "compactness":get_compactness(mesh),
        "convexity": get_convexity(mesh)
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

def get_convexity(mesh:Mesh):
    #(shape volume divided by convex hull volume)
    cvx = ConvexHull(mesh.vertices)
    convexity = mesh.volume()/cvx.volume()
    return convexity

if __name__ == "__main__":
    boxMesh = Box(width=1,height=1,length=1).c("Black").wireframe(True) 
    sphereMesh = Sphere(r=1, res=24, quads=False, c='red', alpha=1.0)
    train = Mesh("..\shapes\Train\D01014.obj")

    print(extract_features(boxMesh))
    print(extract_features(sphereMesh))
    print(extract_features(train))
    