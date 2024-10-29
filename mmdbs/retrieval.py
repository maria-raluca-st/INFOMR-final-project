from scipy.stats import wasserstein_distance as emd
import numpy as np
import pandas as pd

non_hist_features = ['area', 'volume', 'rectangularity', 'compactness', 'convexity', 'eccentricity', 'diameter']

class retriever():
    def __init__(self,feature_df,dropped=["mesh_name", "class"],non_hist_features = ['area', 'volume', 'rectangularity', 'compactness', 'convexity', 'eccentricity', 'diameter']):
        df_features = pd.read_csv(feature_df)
        self.X = df_features.drop(dropped, axis=1)
        hist_features = self.X.columns.difference(non_hist_features)
        self.hist_features = hist_features[~(hist_features.str.contains('center'))]
        self.grouped_columns = hist_features.groupby(hist_features.str[:2])
        self.grouped_index_cols = {k: sorted([self.X.columns.get_loc(c) for c in v]) for k,v in self.grouped_columns.items()}
        self.X[non_hist_features] = (self.X[non_hist_features] - self.X[non_hist_features].mean()) / self.X[non_hist_features].std()


    def hist_distances(self,u, v):
        dist = np.zeros(len(self.grouped_columns))
        for i, cols in enumerate(self.grouped_index_cols.values()):
            dist[i] = emd(u[cols], v[cols])
        return dist


    def dist_func(self,u, v, weights = 1.):
        dist = np.zeros(len(non_hist_features) + len(self.grouped_columns))
        dist[:len(non_hist_features)] = np.linalg.norm(u[:len(non_hist_features)] - v[:len(non_hist_features)])
        dist[len(non_hist_features):] = self.hist_distances(u, v)

        return (weights * dist)

    def retrieve_topk(x, others, k=4, weights=1.):
        dist = np.fromiter((self.dist_func(x, o, weights).sum() for o in others), dtype=x.dtype)
        return dist.argsort()[:k]

if __name__=="main":
    ret = retriever("mesh_features.csv")
