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

import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from normalize import get_center_of_mass, get_eigenvectors
from tqdm.contrib.concurrent import process_map
from normalize import normalize_shape
from vedo import ConvexHull, Mesh

T_RANGE = {  # Theorethical Range of values
    "A3": (0, np.pi),  # Angle Between 3 Points (cos)
    "D1": (0, np.sqrt(3)),  # Worst case scenario very unlikely
    "D2": (0, 3**0.5),  # Widest range of the bounding box
    "D3": (
        0,
        np.sqrt(1 / 2 * (2**0.5)),
    ),  # Triangle on the largest rectangle in unit cube, largest rectangle = 1*sqrt(2)
    "D4": (
        0,
        (1 / 3) ** (1 / 3),
    ),  # Tetrahedon between maximully separated points (0,0,0),(0,1,1),(1,1,0),(1,0,1), we construct a triangle (0,0,0),(0,1,1),(1,1,0) as the base
    # Sides of length sqrt(2), A=1/2*sqrt(2)*sqrt(2). The point 1,0,1 has distance one from this triangle. i.e 1/3*1/2*sqrt(2)*sqrt(2)
}


def extract_features(mesh: Mesh):
    cvx = ConvexHull(mesh.vertices)
    diameterRet = get_diameter(mesh, cvx)
    mesh.fill_holes()
    mesh.triangulate()
    ret = {
        "area": mesh.area(),
        "volume": mesh.volume(),
        "rectangularity": get_rectangularity(mesh),
        "compactness": get_compactness(mesh),
        "convexity": get_convexity(mesh, cvx),
        "eccentricity": get_eccentricity(mesh),
        "diameter": diameterRet[0],
        "diameterPts": diameterRet[1],
        "distributions": get_distributions(mesh),
    }
    return ret


# 2 methods to avoid NaN values
def safe_norm(v):
    norm = np.linalg.norm(v)
    if np.isnan(norm) or np.isinf(norm):
        return 0.0
    return norm


def safe_cross(v1, v2):
    cross_prod = np.cross(v1, v2)
    if np.isnan(cross_prod).any() or np.isinf(cross_prod).any():
        return np.zeros_like(cross_prod)
    return cross_prod


def subsample_vertices(vertices, num_samples=1000, num_vertices=3):
    N = len(vertices)
    k = min(num_samples, N // num_vertices)

    sampled_indices = np.random.choice(N, size=(k, num_vertices), replace=False)

    sampled_groups = vertices[sampled_indices]

    return sampled_groups


def get_surface_area(mesh: Mesh):
    return mesh.area()


def get_rectangularity(mesh: Mesh):
    # How close is the shape (post normalisation to its oriented bounding box)
    # (shape volume divided by OBB volume)
    bbox = mesh.bounds()
    Dx = np.abs(bbox[0] - bbox[1])
    Dy = np.abs(bbox[2] - bbox[3])
    Dz = np.abs(bbox[4] - bbox[5])
    obbVol = Dx * Dy * Dz
    rectangularity = mesh.volume() / obbVol
    return rectangularity


def get_compactness(mesh: Mesh):
    # How close is the shape to a sphere
    return mesh.area() ** 3 / (36 * np.pi * (mesh.volume() ** 2))


def get_convexity(mesh: Mesh, cvx: ConvexHull):
    # (shape volume divided by convex hull volume)
    convexity = mesh.volume() / cvx.volume()
    return convexity


def get_diameter(mesh: Mesh, cvx: ConvexHull, k=500):
    maxD = 0
    maxP = [None, None]
    if len(cvx.vertices) < k:
        subs = cvx.vertices
    else:
        subs = cvx.vertices[np.random.choice(cvx.vertices.shape[0], k, replace=False)]
    for v1 in subs:
        for v2 in cvx.vertices:
            d = np.linalg.norm(v1 - v2)
            if d > maxD:
                maxD = d
                maxP = [v1, v2]
    return maxD, maxP


def get_eccentricity(mesh: Mesh):
    # ratio of largest to smallest eigenvalues of covariance matrix
    _, eigval = get_eigenvectors(mesh)
    mineig = min(eigval)
    maxeig = max(eigval)
    return np.abs(maxeig) / np.abs(mineig)


def get_angle_between_vectors(v1, v2):
    # angle between two vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def get_hist(desc_vals, bins=10, normalize=True, range=(0, 1)):
    hist, bin_edges = np.histogram(desc_vals, bins=bins, density=False, range=range)
    if normalize:
        hist = hist / np.sum(hist)

    # return hist, bin_centers
    return hist, bin_edges[:-1]


def get_distributions(mesh: Mesh, show=True):
    num_samples = 100000
    bins = 30

    A3_vals = calc_A3(mesh, num_samples)
    D1_vals = calc_D1(mesh, num_samples)
    D2_vals = calc_D2(mesh, num_samples)
    D3_vals = calc_D3(mesh, num_samples)
    D4_vals = calc_D4(mesh, num_samples)

    A3_hist, A3_bin_centers = get_hist(A3_vals, bins, T_RANGE["A3"])
    D1_hist, D1_bin_centers = get_hist(D1_vals, bins, T_RANGE["D1"])
    D2_hist, D2_bin_centers = get_hist(D2_vals, bins, T_RANGE["D2"])
    D3_hist, D3_bin_centers = get_hist(D3_vals, bins, T_RANGE["D3"])
    D4_hist, D4_bin_centers = get_hist(D4_vals, bins, T_RANGE["D4"])

    distributions = {
        "A3": (A3_hist, A3_bin_centers),
        "D1": (D1_hist, D1_bin_centers),
        "D2": (D2_hist, D2_bin_centers),
        "D3": (D3_hist, D3_bin_centers),
        "D4": (D4_hist, D4_bin_centers),
    }
    return distributions


# Calculate distance between the center and a random subset, returns list of euclidian distances
# for D1 max num samples = nr of vertices
def calc_D1(mesh: Mesh, num_samples=500):
    center = get_center_of_mass(mesh)
    distances = [safe_norm(vertex - center) for vertex in mesh.vertices]
    return distances


# max num samples  = (nr of vertices)^3
def calc_A3(mesh: Mesh, num_samples=500):
    # A3 for x(num_samples) number of times for each shape
    angles = []
    vertices = mesh.vertices
    sampled_triples = subsample_vertices(vertices, num_samples, num_vertices=3)
    for triple in sampled_triples:
        v1, v2, v3 = triple
        vec1 = v2 - v1
        vec2 = v3 - v1
        # angle = get_angle_between_vectors(vec1, vec2)
        # angles.append(angle)
        if safe_norm(vec1) > 0 and safe_norm(vec2) > 0:
            angle = get_angle_between_vectors(vec1, vec2)
            if not np.isnan(angle) and not np.isinf(angle):
                angles.append(angle)
    return np.array(angles)


# max num samples  = (nr of vertices) ^ 2
def calc_D2(mesh: Mesh, num_samples=500):
    # D2 for x number of times, return concatenated val
    distances = []
    vertices = mesh.vertices
    sampled_duples = subsample_vertices(vertices, num_samples, num_vertices=2)
    for duple in sampled_duples:
        v1, v2 = duple
        distance = safe_norm(v1 - v2)
        if distance > 0 and not np.isnan(distance) and not np.isinf(distance):
            distances.append(distance)
    return np.array(distances)


# max num samples  = (nr of vertices)^3
def calc_D3(mesh: Mesh, num_samples=500):
    areas = []
    vertices = mesh.vertices
    sampled_triples = subsample_vertices(vertices, num_samples, num_vertices=3)
    for triple in sampled_triples:
        v1, v2, v3 = triple
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        if area > 0 and not np.isnan(area) and not np.isinf(area):
            areas.append(np.sqrt(area))
    return np.array(areas)


# max num samples  = (nr of vertices)^4
def calc_D4(mesh: Mesh, num_samples=500):
    volumes = []
    vertices = mesh.vertices
    sampled_quadruples = subsample_vertices(vertices, num_samples, num_vertices=4)
    for q in sampled_quadruples:
        v1, v2, v3, v4 = q
        # volume of tetrahedron
        volume = np.abs(np.dot((v4 - v1), np.cross(v2 - v1, v3 - v1))) / 6
        if volume > 0 and not np.isnan(volume) and not np.isinf(volume):
            volumes.append(np.cbrt(volume))
    return np.array(volumes)


def process_vedo_mesh(mesh: Mesh):
    features = extract_features(mesh)

    feature_row = {
        "area": features["area"],
        "volume": features["volume"],
        "rectangularity": features["rectangularity"],
        "compactness": features["compactness"],
        "convexity": features["convexity"],
        "eccentricity": features["eccentricity"],
        "diameter": features["diameter"],
    }  # Add histograms
    for dist_name, dist_values in features["distributions"].items():
        histogram_values = dist_values[0]  # Histogram
        # Store histogram values
        for bin_idx, bin_value in enumerate(histogram_values):
            feature_row[f"{dist_name}_bin_{bin_idx}"] = bin_value

    df_feature = pd.DataFrame([feature_row])

    return df_feature


def process_mesh(shape_directory, original_path, normalize=False, output_directory=""):
    class_name = original_path.parent.name
    mesh_file = str(shape_directory / class_name / original_path.name)
    #print(mesh_file)
    mesh = Mesh(mesh_file)
    
    if(normalize):
        output_file = str(output_directory / class_name / original_path.name)
        mesh = normalize_shape(mesh)
        mesh.write(output_file)


    features = extract_features(mesh)

    feature_row = {
        "mesh_name": original_path.name,
        "class": class_name,
        "area": features["area"],
        "volume": features["volume"],
        "rectangularity": features["rectangularity"],
        "compactness": features["compactness"],
        "convexity": features["convexity"],
        "eccentricity": features["eccentricity"],
        "diameter": features["diameter"],
    }

    # Add histograms
    for dist_name, dist_values in features["distributions"].items():
        histogram_values = dist_values[0]  # Histogram
        bin_centers = dist_values[1]  # Bin centers

        # Store histogram values
        for bin_idx, bin_value in enumerate(histogram_values):
            feature_row[f"{dist_name}_bin_{bin_idx}"] = bin_value

        # Store bin centers
        for bin_idx, bin_center in enumerate(bin_centers):
            feature_row[f"{dist_name}_bin_center_{bin_idx}"] = bin_center

    return feature_row


def extract_dataset_features_from_shapes(
    manifest, shape_directory="../normshapes", output_file="subset_mesh_features.csv", normalize=False, output_directory=""
):
    feature_data = []
    skipped = 0
    shape_directory = Path(shape_directory)
    output_directory = Path(output_directory)
    feature_data = process_map(
        partial(process_mesh, shape_directory, normalize=normalize, output_directory=output_directory), manifest["Path"].apply(Path)
    )

    feature_df = pd.DataFrame(feature_data)
    target_directory = os.getcwd()

    output_path = os.path.join(target_directory, output_file)
    feature_df.to_csv(output_path, index=False)
    print(f"Feature extraction complete. Features saved to {output_path}.")
    print("Skipped meshes count: " + str(skipped))

if __name__ == "__main__":
    # Uncomment these lines if you want to run retrieval on the subset
    # df_manifest_subset = pd.read_csv("./subset_shape_manifest.csv")
    # df_manifest_subset = df_manifest_subset[df_manifest_subset["ReturnCode"] == 0]
    df_manifest = pd.read_csv("./shape_manifest.csv")
    df_manifest = df_manifest[df_manifest["ReturnCode"] == 0]
    """
    extract_dataset_features_from_shapes(
        df_manifest,
        shape_directory="../shapes",
        output_file="mesh_features.csv",
        normalize=True,
        output_directory="../normshapes",

    )
    """
    
    extract_dataset_features_from_shapes(
        df_manifest,
        shape_directory="../normshapes",
        output_file="mesh_features.csv",
        normalize=False,
        output_directory="..normshapes",
    )
    
    

    
