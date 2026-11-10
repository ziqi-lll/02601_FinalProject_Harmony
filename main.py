"""Run Preprocessing -> Harmony -> Evaluation with 7 focused outputs.

Run: python main.py
Only figures and metrics needed for a concise Harmony2 replication report are saved.
No raw data are modified. See README.md for output details.
"""

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from preprocessing import (
    load_input, calculate_qc_metrics, filter_cells, filter_genes,
    normalize_log1p, select_highly_variable_genes, calculate_pca,
)
from evaluation import (
    qc_summary, benchmark_methods, compare_embeddings, mixing_metrics,
    make_umaps, plot_embedding_comparison, plot_diagnostics,
    evaluate_simulation,
)
from harmony import objective_converged


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "LV_raw.h5ad"
MITO_PATH = BASE_DIR / "data" / "human_mt_gene_ids.txt"
OUTPUT_DIR = BASE_DIR / "results"
SOURCE_KEY = "hubmap_id"

PREPROCESSING = {
    "min_genes": 200, "max_mito": 20, "min_cells": 3,
    "target_sum": 10000.0, "n_top_genes": 2000, "n_pcs": 30,
    "clip_value": 10.0,
}
HARMONY = {
    "theta": 2.0, "lamb": None, "sigma": 0.1, "nclust": 50,
    "tau": 0.0, "block_size": 0.05,
    "max_iter_harmony": 20, "max_iter_kmeans": 4,
    "epsilon_cluster": 1e-3, "epsilon_harmony": 1e-2,
    "alpha": 0.2, "batch_prop_cutoff": 1e-5, "random_state": 0,
}
BENCHMARK_REPEATS = 3


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Input QC", flush=True)
    adata, _ = load_input(INPUT_PATH, SOURCE_KEY)
    input_shape = list(adata.shape)
    mito_ids = set(MITO_PATH.read_text().splitlines())
    adata = calculate_qc_metrics(adata, mito_ids)
    before_obs = adata.obs.copy()

    print("[2/6] Preprocessing and PCA", flush=True)
    preprocessing_start = perf_counter()
    adata = filter_cells(adata, PREPROCESSING["min_genes"], PREPROCESSING["max_mito"])
    after_cell_shape = list(adata.shape)
    counts = qc_summary(before_obs, adata.obs, SOURCE_KEY)
    counts.to_csv(OUTPUT_DIR / "qc_by_source.csv")
    adata = filter_genes(adata, PREPROCESSING["min_cells"])
    after_gene_shape = list(adata.shape)
    adata = normalize_log1p(adata, PREPROCESSING["target_sum"])
    adata = select_highly_variable_genes(adata, SOURCE_KEY, PREPROCESSING["n_top_genes"])
    adata = calculate_pca(adata, PREPROCESSING["n_pcs"], PREPROCESSING["clip_value"], HARMONY["random_state"])
    preprocessing_seconds = perf_counter() - preprocessing_start

    labels = adata.obs[SOURCE_KEY].copy()
    metadata = adata.obs[[SOURCE_KEY]].copy()
    pcs = adata.obsm["X_pca"].astype(np.float64)
    if HARMONY["nclust"] > len(pcs):
        raise ValueError("Reduce nclust for this smaller input.")
    print(f"  {input_shape} -> {after_cell_shape} -> {after_gene_shape}; PCA {pcs.shape}", flush=True)

    print("[3/6] Run and benchmark both Harmony implementations", flush=True)
    ours, reference, timing = benchmark_methods(
        pcs, metadata, SOURCE_KEY, HARMONY, BENCHMARK_REPEATS
    )
    timing_summary = timing.groupby("method")["seconds"].agg(
        ["median", "min", "max", "mean", "std"]
    )

    print("[4/6] Numerical agreement and batch mixing", flush=True)
    embeddings = {
        "Uncorrected": pcs,
        "Python Harmony2": ours["Z_corr"],
        "harmonypy 2.1.0": reference.Z_corr,
    }
    agreement, per_pc = compare_embeddings(ours["Z_corr"], reference.Z_corr)
    mixing = pd.DataFrame([
        {"method": name, **mixing_metrics(values, labels)}
        for name, values in embeddings.items()
    ])
    mixing.to_csv(OUTPUT_DIR / "source_mixing_metrics.csv", index=False)

    print("[5/6] Embedding comparison and diagnostics", flush=True)
    umaps = make_umaps(embeddings, HARMONY["random_state"])
    plot_embedding_comparison(
        embeddings, umaps, labels, OUTPUT_DIR / "embedding_comparison.png"
    )
    plot_diagnostics(
        ours, reference, timing, per_pc,
        OUTPUT_DIR / "diagnostics.png",
    )

    print("[6/6] Known-truth simulation and summary", flush=True)
    simulation = evaluate_simulation(HARMONY, OUTPUT_DIR)
    reference_objective = reference.objective_harmony
    reference_converged = len(reference_objective) >= 2 and objective_converged(
        reference_objective[-2], reference_objective[-1], HARMONY["epsilon_harmony"]
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_shape": input_shape,
        "after_cell_qc_shape": after_cell_shape,
        "after_gene_filter_shape": after_gene_shape,
        "pca_shape": list(pcs.shape),
        "mitochondrial_genes_in_input": int(adata.var["is_mito"].sum()),
        "highly_variable_genes": int(adata.var["highly_variable"].sum()),
        "pca_cumulative_variance": float(adata.uns["pca"]["variance_ratio"].sum()),
        "preprocessing_parameters": PREPROCESSING,
        "harmony_parameters": HARMONY,
        "python_converged": ours["converged"],
        "reference_converged": bool(reference_converged),
        "python_iterations": ours["n_iter"],
        "reference_iterations": len(reference.kmeans_rounds),
        "python_vs_reference": agreement,
        "runtime_medians_seconds": timing_summary["median"].to_dict(),
        "preprocessing_seconds": preprocessing_seconds,
        "simulation": simulation.to_dict(orient="records"),
        "total_pipeline_seconds": perf_counter() - started,
        "interpretation_limits": [
            "Donor and source are confounded; better mixing is not proof of biological preservation.",
            "No verified cell-type labels are available for the real data.",
            "Separate UMAP fits have no directly comparable absolute coordinates.",
            "Implementations may differ numerically because of random number generators and precision.",
        ],
    }
    write_json(OUTPUT_DIR / "run_summary.json", summary)
    print("\nCompleted. 7 focused output files in:", OUTPUT_DIR, flush=True)
    print(mixing.to_string(index=False), flush=True)
    print("\nPython vs reference:", json.dumps(agreement, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
