from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as emd
from tqdm import tqdm

NON_HIST_FEATURES = [
    "area",
    "volume",
    "rectangularity",
    "compactness",
    "convexity",
    "eccentricity",
    "diameter",
]



class RetrievalEngine:
    def __init__(
        self,
        feature_path: Path = Path("mesh_features.csv"),
        dropped=["mesh_name", "class"],
        non_hist_features=NON_HIST_FEATURES,
        hist_norm_path: Path = Path("histogram_zscore.npz")
    ):
        # Load features and separate metadata
        df_features = pd.read_csv(feature_path)
        X = df_features.drop(dropped, axis=1)
        self.metadata = df_features[dropped]

        # Identify and Z-score standardise Non-Histogram (scalar) features
        self.non_hist_features = non_hist_features
        X[non_hist_features] = (
            X[non_hist_features] - X[non_hist_features].mean()
        ) / X[non_hist_features].std()
        
        # Identify Histogram features, remove auxiliary columns 
        hist_features = X.columns.difference(non_hist_features)
        self.hist_features = hist_features[~(hist_features.str.contains("center"))]

        # Separate histogram features into distinct histograms (A3, D1-D4)
        self.grouped_columns = hist_features.groupby(hist_features.str[:2])
        self.grouped_index_cols = {
            k: sorted([X.columns.get_loc(c) for c in v])
            for k, v in self.grouped_columns.items()
        }
        
        self.X = X.to_numpy()

        # TBD: weighing for the histogram distances, placeholder values
        # self.mu = np.zeros(len(self.grouped_index_cols.keys()))
        # self.sigma = np.ones(len(self.grouped_index_cols.keys()))
        if hist_norm_path.exists():
            arr = np.load(hist_norm_path)
            self.mu = arr['mu']
            self.sigma = arr['sigma']
        else:
            self.mu, self.sigma = self._compute_hist_distance_stats()
            np.savez(hist_norm_path, mu=self.mu, sigma=self.sigma)

    def _compute_hist_distance_stats(self):
        """Compute mean and standard deviation of distances for each histogram group."""

        num_histograms = len(self.grouped_index_cols)
        mu = np.zeros(num_histograms)
        sigma = np.ones(num_histograms)

        total = factorial(self.X.shape[0]) / (2 * factorial(self.X.shape[0] - 2))

        for i, (hist_name, cols) in enumerate(self.grouped_index_cols.items()):
            # pairwise distances 
            distances = []
        
            for u, v in tqdm(combinations(self.X, 2), total=total):
                distances.append(emd(u[cols], v[cols]))
            distances = np.array(distances)
            # mean and std for this histogram's distances
            mu[i] = distances.mean()
            sigma[i] = distances.std()
        
        return mu, sigma

    def hist_distances(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Calculates distance weighted EMD between feature vectors u and v, where both contain multiple histograms.


        Args:
            u (np.ndarray): Query
            v (np.ndarray): Sample

        Returns:
            np.ndarray: EMD distances
        """
        dist = np.zeros(len(self.grouped_columns))
        for i, cols in enumerate(self.grouped_index_cols.values()):
            dist[i] = emd(u[cols], v[cols])
        return (dist - self.mu) / self.sigma

    def dist_func(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Combination of L2 for scalar features and EMD for histogram features.

        Args:
            u (np.ndarray): Query
            v (np.ndarray): Sample

        Returns:
            np.ndarray: L2 followed by weighted EMD distances
        """
        dist = np.zeros(len(self.non_hist_features) + len(self.grouped_columns))
        dist[: len(self.non_hist_features)] = np.linalg.norm(
            u[: len(self.non_hist_features)] - v[: len(self.non_hist_features)]
        )
        dist[len(self.non_hist_features) :] = self.hist_distances(u, v)

        return dist

    def retrieve_topk(self, x, k=4):
        dist = np.fromiter(
            (self.dist_func(x, o).sum() for o in self.X), dtype=x.dtype
        )
        topk_idx = dist.argsort()[:k]
        return self.metadata.iloc[topk_idx], dist[topk_idx]
    
    def retrieve_topr(self, x, r=5):
        dist = np.fromiter((self.dist_func(x, o).sum() for o in self.X), dtype=x.dtype)
        topr_idx = np.argwhere(dist < r).flatten()
        return self.metadata.iloc[topr_idx], dist[topr_idx]
 

if __name__ == "__main__":
    ret = RetrievalEngine("mesh_features.csv")
    meta, dist = ret.retrieve_topk(ret.X[56])
    meta = meta.copy()
    meta['dist'] = dist
    print("========= Top k=4 ===========")
    print(meta, end="\n\n")
    meta, dist = ret.retrieve_topr(ret.X[0])
    meta = meta.copy()
    meta['dist'] = dist
    print("========= Top r=5 ===========")
    print(meta, end="\n\n")

    print("Distances between objects 56 vs 1122")
    df_dist = pd.Series(ret.dist_func(ret.X[75], ret.X[1122]), index=NON_HIST_FEATURES + list(ret.grouped_index_cols.keys()))
    print(df_dist, end="\n\n")


    print("When u = v, dist is approx the standardization values:")
    print((-ret.mu / ret.sigma).sum(), end="\n\n")

