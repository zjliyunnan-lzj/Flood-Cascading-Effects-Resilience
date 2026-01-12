# -*- coding: utf-8 -*-
"""
Permutation-test experiment for interdependence among centrality metrics.

Input (CSV/TSV/Excel) recommended columns:
node_id | period | degree | betweenness | closeness | eigenvector

Outputs (in OUTPUT_DIR):
1) centrality_dependency_permutation.xlsx
   - <period>_LONG: pairwise results (rho_obs, p_perm)
   - <period>_RHO: rho_obs matrix
   - <period>_P  : p_perm matrix
   - SUMMARY     : count of significant pairs per period
2) heatmap_rho_obs_<period>.png
3) heatmap_p_perm_<period>.png
4) heatmap_rho_obs_ALL.png, heatmap_p_perm_ALL.png (pooled)
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =========================
# CONFIG — edit only here
# =========================
FILEPATH = r"file.csv"  # <-- replace with your CSV path
SHEET_NAME = None  # required only for Excel, e.g., "Sheet1"

OUTPUT_DIR = r"folder"

# Column names (modify here if your CSV headers differ)
COL_NODE = "node_id"
COL_PERIOD = "period"  # if absent, the script will treat data as single-period "ALL"
METRICS = ["degree", "betweenness", "closeness", "eigenvector"]

# Permutation test parameters
N_PERM = 2000          # typically 1000–5000; 1000 is sufficient for revision
RANDOM_STATE = 42
ALPHA = 0.05

# Figure settings
DPI = 220

# Optional: if CSV encoding is GBK (common for Excel exports), set to "gbk"
CSV_ENCODING = None  # None / "utf-8" / "gbk"
CSV_SEP = ","        # "," for CSV; ";" for semicolon; "\t" for TSV


# =========================
# I/O helpers
# =========================
def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\nPlease check FILEPATH."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path, encoding=CSV_ENCODING, sep=CSV_SEP)
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path, encoding=CSV_ENCODING, sep="\t")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=SHEET_NAME)

    raise ValueError(f"Unsupported file type: {ext}")


def safe_excel_path(out_dir: str, filename: str) -> str:
    """
    Avoid PermissionError:
    1) If the old file exists and is not locked -> delete and overwrite
    2) If the file is locked by Excel -> append a timestamp
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, filename)
    if not os.path.exists(base):
        return base
    try:
        os.remove(base)
        return base
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        return os.path.join(out_dir, f"{name}_{ts}{ext}")


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Period column is optional
    if COL_PERIOD not in df.columns:
        df = df.copy()
        df[COL_PERIOD] = "ALL"

    required = [COL_NODE, COL_PERIOD] + METRICS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join([f"- {c}" for c in missing])
            + "\n\nPlease check CSV headers or modify "
            "COL_NODE / COL_PERIOD / METRICS in the script."
        )

    df = df.copy()
    df[COL_NODE] = df[COL_NODE].astype(str)
    df[COL_PERIOD] = df[COL_PERIOD].astype(str)

    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).copy()

    # Remove infinite values
    for c in METRICS:
        df = df[np.isfinite(df[c])]

    return df


# =========================
# Permutation test
# =========================
def permutation_test_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator
):
    """
    Returns:
      rho_obs: observed Spearman rho
      p_perm : two-sided permutation p-value
    """
    rho_obs, _ = spearmanr(x, y)

    rho_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        rho_perm[i], _ = spearmanr(x, y_perm)

    p_perm = (np.sum(np.abs(rho_perm) >= np.abs(rho_obs)) + 1) / (n_perm + 1)
    return float(rho_obs), float(p_perm)


# =========================
# Plot
# =========================
def plot_heatmap(mat: pd.DataFrame, title: str, outfile: str, fmt: str = ".2f"):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=DPI)
    im = ax.imshow(mat.values)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            txt = "NA" if np.isnan(v) else format(v, fmt)
            ax.text(j, i, txt, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# =========================
# Run for one period
# =========================
def run_for_one_period(
    g: pd.DataFrame,
    period: str,
    n_perm: int,
    seed: int
):
    rng = np.random.default_rng(seed)

    rho_mat = pd.DataFrame(index=METRICS, columns=METRICS, dtype=float)
    p_mat = pd.DataFrame(index=METRICS, columns=METRICS, dtype=float)

    for m in METRICS:
        rho_mat.loc[m, m] = 1.0
        p_mat.loc[m, m] = 0.0

    rows = []
    for m1, m2 in itertools.combinations(METRICS, 2):
        x = g[m1].to_numpy()
        y = g[m2].to_numpy()
        rho_obs, p_perm = permutation_test_spearman(
            x, y, n_perm=n_perm, rng=rng
        )

        rho_mat.loc[m1, m2] = rho_obs
        rho_mat.loc[m2, m1] = rho_obs
        p_mat.loc[m1, m2] = p_perm
        p_mat.loc[m2, m1] = p_perm

        rows.append({
            "period": period,
            "metric_1": m1,
            "metric_2": m2,
            "n_nodes": len(g),
            "rho_obs": rho_obs,
            "p_perm": p_perm,
            "n_perm": n_perm,
            "random_state": seed
        })

    long_df = (
        pd.DataFrame(rows)
        .sort_values(["metric_1", "metric_2"])
        .reset_index(drop=True)
    )
    return long_df, rho_mat, p_mat


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_table(FILEPATH)
    df = validate_and_clean(df)

    periods = sorted(df[COL_PERIOD].unique().tolist())
    out_xlsx = safe_excel_path(
        OUTPUT_DIR,
        "centrality_dependency_permutation.xlsx"
    )

    summary_rows = []
    all_long = []

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for i, period in enumerate(periods):
            g = (
                df[df[COL_PERIOD] == period][[COL_NODE] + METRICS]
                .dropna()
                .copy()
            )

            seed = RANDOM_STATE + i
            long_df, rho_mat, p_mat = run_for_one_period(
                g, period, N_PERM, seed
            )

            long_df.to_excel(writer, sheet_name=f"{period}_LONG", index=False)
            rho_mat.to_excel(writer, sheet_name=f"{period}_RHO")
            p_mat.to_excel(writer, sheet_name=f"{period}_P")

            n_sig = int((long_df["p_perm"] < ALPHA).sum())
            summary_rows.append({
                "period": period,
                "n_nodes": len(g),
                "n_pairs": len(long_df),
                f"n_sig_p<{ALPHA}": n_sig
            })
            all_long.append(long_df)

            plot_heatmap(
                rho_mat,
                f"Permutation-test Spearman rho_obs (period={period})",
                os.path.join(
                    OUTPUT_DIR,
                    f"heatmap_rho_obs_{period}.png"
                ),
                fmt=".2f"
            )
            plot_heatmap(
                p_mat,
                f"Permutation-test p_perm (period={period})",
                os.path.join(
                    OUTPUT_DIR,
                    f"heatmap_p_perm_{period}.png"
                ),
                fmt=".3g"
            )

        combined = (
            pd.concat(all_long, ignore_index=True)
            if all_long else pd.DataFrame()
        )
        combined.to_excel(
            writer, sheet_name="ALL_LONG", index=False
        )

        pooled = df[[COL_NODE] + METRICS].dropna().copy()
        pooled_long, pooled_rho, pooled_p = run_for_one_period(
            pooled,
            "ALL_POOLED",
            N_PERM,
            RANDOM_STATE + 999
        )

        pooled_long.to_excel(
            writer, sheet_name="ALL_POOLED_LONG", index=False
        )
        pooled_rho.to_excel(
            writer, sheet_name="ALL_POOLED_RHO"
        )
        pooled_p.to_excel(
            writer, sheet_name="ALL_POOLED_P"
        )

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values("period")
            .reset_index(drop=True)
        )
        summary_df.to_excel(
            writer, sheet_name="SUMMARY", index=False
        )

    plot_heatmap(
        pooled_rho,
        "Permutation-test Spearman rho_obs (ALL pooled)",
        os.path.join(
            OUTPUT_DIR,
            "heatmap_rho_obs_ALL.png"
        ),
        fmt=".2f"
    )
    plot_heatmap(
        pooled_p,
        "Permutation-test p_perm (ALL pooled)",
        os.path.join(
            OUTPUT_DIR,
            "heatmap_p_perm_ALL.png"
        ),
        fmt=".3g"
    )

    print("Permutation test finished.")
    print("Excel saved to:", os.path.abspath(out_xlsx))
    print("Figures saved in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
