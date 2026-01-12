import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Plot settings
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

"""
This script performs t-SNE dimensionality reduction and K-means clustering.
Raw data are not included due to confidentiality restrictions.
"""

# 1) Load Excel data
file_path = r"file.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# 2) Check and clean data
if df.isnull().sum().sum() > 0:
    print("Missing values detected. Filled using median values.")
    df = df.fillna(df.median(numeric_only=True))

# Keep the first 5 columns (e.g., ID, coordinates) and use the remaining columns as features
columns_keep = df.columns[:5]
columns_to_use = df.columns[5:]

X = df[columns_to_use]

# 3) t-SNE (features -> 2D)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# 4) Evaluate candidate K using silhouette score (for reference only)
K_range = range(2, 10)
sil_scores = []

for k in K_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = kmeans_tmp.fit_predict(X_tsne)
    sil_scores.append(silhouette_score(X_tsne, labels_tmp))

plt.figure(figsize=(6, 4))
plt.plot(list(K_range), sil_scores, marker="o", linestyle="-")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score across K")
plt.grid(True)
plt.show()

# 5) Fit final K-means with fixed K
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_tsne)

sil = silhouette_score(X_tsne, clusters)
ch = calinski_harabasz_score(X_tsne, clusters)
print(f"Silhouette score: {sil:.4f}; Calinski-Harabasz index: {ch:.2f}")

# 6) Build output table
df_out = df[columns_keep].copy()
df_out["TSNE1"] = X_tsne[:, 0]
df_out["TSNE2"] = X_tsne[:, 1]
df_out["Cluster"] = clusters

# 7) Plot t-SNE + clustering result
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="TSNE1",
    y="TSNE2",
    hue="Cluster",
    palette="tab10",
    s=50,
    data=df_out
)
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.title("t-SNE projection with K-means clusters")
plt.legend(title="Cluster")

plt.xlim(-80, 80)
plt.ylim(-50, 50)
plt.xticks(range(-80, 81, 20))
plt.yticks(range(-50, 51, 10))

figure_path = r"file.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

# 8) Export results
output_path = r"file.xlsx"
df_out.to_excel(output_path, index=False)

print(f"Results exported to: {output_path}")
print(f"Figure saved to: {figure_path}")
