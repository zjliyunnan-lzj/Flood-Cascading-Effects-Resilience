import os
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        required=True,
        help='Input mapping like: "2005=2005.xlsx,2010=2010.xlsx,2015=2015.xlsx,2020=2020.xlsx"',
    )
    p.add_argument("--out-dir", default="outputs", help="Output directory")
    p.add_argument("--k", type=int, default=4, help="KMeans n_clusters")
    p.add_argument("--n-init", type=int, default=10, help="KMeans n_init")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--n-runs", type=int, default=100, help="Number of seeds for CI")
    p.add_argument("--tsne-seed", type=int, default=42, help="t-SNE random_state")
    p.add_argument("--base-seed", type=int, default=42, help="Base KMeans random_state for point estimate")
    p.add_argument("--start-col", type=int, default=5, help="0-based index: feature columns start at this position")
    return p.parse_args()


def parse_inputs(inputs_str: str) -> dict[str, str]:
    items = [x.strip() for x in inputs_str.split(",") if x.strip()]
    mapping: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid inputs item: {it}. Expected YEAR=PATH.")
        year, path = it.split("=", 1)
        year = year.strip()
        path = path.strip().strip('"').strip("'")
        if not year or not path:
            raise ValueError(f"Invalid inputs item: {it}.")
        mapping[year] = path
    if not mapping:
        raise ValueError("No valid inputs parsed.")
    return mapping


def median_impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df[cols].isnull().sum().sum() > 0:
        df = df.copy()
        df[cols] = df[cols].apply(lambda s: s.fillna(s.median()))
    return df


def compute_metrics_for_year(
    year: str,
    file_path: str,
    k: int,
    n_init: int,
    perplexity: float,
    n_runs: int,
    tsne_seed: int,
    base_seed: int,
    start_col: int,
) -> dict:
    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] <= start_col:
        raise ValueError(f"{year}: start_col={start_col} exceeds available columns={df.shape[1]}")

    cols_use = list(df.columns[start_col:])
    df = median_impute(df, cols_use)

    tsne = TSNE(n_components=2, random_state=tsne_seed, perplexity=perplexity)
    data_tsne = tsne.fit_transform(df[cols_use])

    km_base = KMeans(n_clusters=k, random_state=base_seed, n_init=n_init)
    labels_base = km_base.fit_predict(data_tsne)

    ss_point = float(silhouette_score(data_tsne, labels_base))
    ch_point = float(calinski_harabasz_score(data_tsne, labels_base))

    ss_list = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(data_tsne)
        ss_list.append(silhouette_score(data_tsne, labels))

    ss_arr = np.asarray(ss_list, dtype=float)
    ss_ci_low, ss_ci_high = np.quantile(ss_arr, [0.025, 0.975])

    return {
        "year": year,
        "k": int(k),
        "tsne_seed": int(tsne_seed),
        "base_seed": int(base_seed),
        "n_init": int(n_init),
        "perplexity": float(perplexity),
        "n_runs": int(n_runs),
        "ss_point": ss_point,
        "ss_ci_low": float(ss_ci_low),
        "ss_ci_high": float(ss_ci_high),
        "ch_point": ch_point,
        "ss_95ci_text": f"{ss_point:.4f} [{ss_ci_low:.4f}, {ss_ci_high:.4f}]",
        "input_file": os.path.basename(file_path),
        "n_rows": int(df.shape[0]),
        "n_features": int(len(cols_use)),
    }


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.inputs)

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for year, path in inputs.items():
        rows.append(
            compute_metrics_for_year(
                year=year,
                file_path=path,
                k=args.k,
                n_init=args.n_init,
                perplexity=args.perplexity,
                n_runs=args.n_runs,
                tsne_seed=args.tsne_seed,
                base_seed=args.base_seed,
                start_col=args.start_col,
            )
        )

    table = pd.DataFrame(rows).sort_values("year")

    out_xlsx = os.path.join(args.out_dir, "table5_ss_with_95ci.xlsx")
    out_csv = os.path.join(args.out_dir, "table5_ss_with_95ci.csv")

    table.to_excel(out_xlsx, index=False)
    table.to_csv(out_csv, index=False, encoding="utf-8")

    print(out_csv)


if __name__ == "__main__":
    main()
