import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

"""
This script evaluates K-means clustering stability using ARI (Adjusted Rand Index)
under different random initializations, for multiple periods.
Raw data are not included due to confidentiality restrictions.
"""

# Directory containing input files (update as needed)
folder = r"folder"

# Input files per period (update as needed)
files = {
    "2005": "2005_cluster.xlsx",
    "2010": "2010_cluster.xlsx",
    "2015": "2015_cluster.xlsx",
    "2020": "2020_cluster.xlsx",
}

# Optional ID column name
ID_COL = "FID"

# --- Canonical (English) feature names used inside the code ---
FEATURES = [
    "index1", "index2", "index3", "index4", "index5", "index6", "index7",
    "index8", "index9", "index10", "index11",
    "index12", "index13", "index14"
]

def read_X(filepath, features=FEATURES, id_col=ID_COL):
    df = pd.read_excel(filepath)

    # Clean column name whitespace
    df.columns = df.columns.astype(str).str.strip()

    # Drop ID column if present
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    # Check whether all feature columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(filepath)} is missing feature columns: {missing}\n"
            f"Current columns: {list(df.columns)}"
        )

    X = df[features].to_numpy()
    return X

def kmeans_stability_ari(X, k=4, n_repeat=100, base_seed=0):
    X_std = StandardScaler().fit_transform(X)

    km_ref = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        random_state=base_seed
    )
    labels_ref = km_ref.fit_predict(X_std)

    aris = []
    for seed in range(1, n_repeat + 1):
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=50,
            random_state=seed
        )
        labels = km.fit_predict(X_std)
        aris.append(adjusted_rand_score(labels_ref, labels))

    aris = np.array(aris)
    return {
        "n_samples": int(X.shape[0]),
        "ARI_mean": float(aris.mean()),
        "ARI_std": float(aris.std(ddof=1)),
        "ARI_min": float(aris.min()),
        "ARI_p05": float(np.quantile(aris, 0.05)),
    }

results = []
for period, fname in files.items():
    path = os.path.join(folder, fname)
    X = read_X(path)
    res = kmeans_stability_ari(X, k=4, n_repeat=100, base_seed=0)
    res["Period"] = period
    results.append(res)

df_results = pd.DataFrame(results).set_index("Period")
print(df_results[["n_samples", "ARI_mean", "ARI_std", "ARI_min", "ARI_p05"]])