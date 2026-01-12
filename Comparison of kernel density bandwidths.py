import rasterio
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ========= 1. Paths =========
base_dir = r"folder"

raster_files = {
    "0.5h": f"{base_dir}\\file-0h.tif",
    "1.0h": f"{base_dir}\\file-h.tif",
    "1.5h": f"{base_dir}\\file-1h.tif",
}

# ========= 2. Read rasters =========
rasters = {}
nodata_value = None

for key, path in raster_files.items():
    with rasterio.open(path) as src:
        rasters[key] = src.read(1)
        if nodata_value is None:
            nodata_value = src.nodata

# ========= 3. Valid pixel mask =========
mask = np.ones_like(next(iter(rasters.values())), dtype=bool)

for r in rasters.values():
    mask &= ~np.isnan(r)
    if nodata_value is not None:
        mask &= (r != nodata_value)

# ========= 4. Correlation comparison =========
def compare(r1, r2, name1, name2):
    v1 = r1[mask].ravel()
    v2 = r2[mask].ravel()

    pearson, p_p = pearsonr(v1, v2)
    spearman, p_s = spearmanr(v1, v2)

    print(f"\n{name1} vs {name2}")
    print(f"  Pearson r   = {pearson:.4f} (p = {p_p:.2e})")
    print(f"  Spearman rho = {spearman:.4f} (p = {p_s:.2e})")

# ========= 5. Pairwise comparisons =========
compare(rasters["1.0h"], rasters["0.5h"], "1.0h", "0.5h")
compare(rasters["1.0h"], rasters["1.5h"], "1.0h", "1.5h")
compare(rasters["0.5h"], rasters["1.5h"], "0.5h", "1.5h")
