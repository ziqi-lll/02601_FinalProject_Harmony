"""Input, quality control, gene selection, normalization, and PCA.

Expression matrices stay sparse until we scale the selected HVGs for PCA.
The input must contain nonnegative, unlogged count-like values in adata.X.
Salmon estimates can be fractional; they are not rounded.
"""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA


def load_input(file_path, source_key):
    """Read one h5ad and return (AnnData, aligned source labels)."""
    adata = ad.read_h5ad(file_path)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("The input expression matrix is empty.")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell IDs and gene IDs must be unique.")
    if source_key not in adata.obs:
        raise ValueError(f"Source column not found: {source_key}")
    source_labels = adata.obs[source_key].copy()
    if source_labels.isna().any():
        raise ValueError("Some cells have missing source labels.")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("X must contain finite, nonnegative count-like values.")
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    adata.X.sum_duplicates()
    adata.X.eliminate_zeros()
    return adata, source_labels


def calculate_qc_metrics(adata, mito_gene_ids):
    """Annotate QC on the original count matrix; change metadata only.

    mito_gene_ids is the supplied set of 37 human mitochondrial Ensembl IDs,
    without version suffixes. hugo_symbol.startswith('MT-') is incomplete
    for LV_raw.h5ad and must not be used for this dataset.
    """
    x = adata.X
    total_counts = np.asarray(x.sum(axis=1)).ravel()
    n_genes = np.asarray((x > 0).sum(axis=1)).ravel()
    gene_ids = adata.var_names.str.split(".").str[0]
    is_mito = gene_ids.isin(mito_gene_ids)
    if not is_mito.any():
        raise ValueError("No mitochondrial IDs matched. Check gene identifiers.")
    mito_counts = np.asarray(x[:, is_mito].sum(axis=1)).ravel()
    pct_mito = np.full(adata.n_obs, np.nan)
    valid = total_counts > 0
    pct_mito[valid] = mito_counts[valid] / total_counts[valid] * 100
    adata.var["is_mito"] = is_mito
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes"] = n_genes
    adata.obs["pct_mito"] = pct_mito
    return adata


def filter_cells(adata, min_genes=200, max_mito=20):
    """Keep nonempty cells with >= min_genes and < max_mito percent."""
    keep = (
        (adata.obs["total_counts"] > 0)
        & (adata.obs["n_genes"] >= min_genes)
        & (adata.obs["pct_mito"] < max_mito)
    )
    filtered = adata[keep, :].copy()
    if filtered.n_obs == 0:
        raise ValueError("No cells remain. Check the QC thresholds.")
    return filtered


def filter_genes(adata, min_cells=3):
    """Keep genes detected (>0) in at least min_cells retained cells."""
    if not isinstance(min_cells, (int, np.integer)) or min_cells < 1:
        raise ValueError("min_cells must be a positive integer.")
    n_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep = n_cells >= min_cells
    if not keep.any():
        raise ValueError("No genes remain. Check min_cells.")
    filtered = adata[:, keep].copy()
    filtered.var["n_cells"] = n_cells[keep]
    return filtered


def normalize_log1p(adata, target_sum=1e4):
    """Save retained-gene counts, normalize each cell's total, then log1p.

    After this call, X is log1p(normalized counts) and layers['counts'] holds
    unnormalized counts for the same cells/genes. QC obs columns still refer
    to the original pre-gene-filter counts. Never call this function twice.
    """
    if "counts" in adata.layers:
        raise ValueError("counts layer already exists; normalization may already be done.")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be positive and finite.")
    x = sparse.csr_matrix(adata.X, dtype=np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("A cell has zero counts after gene filtering.")
    adata.layers["counts"] = x.copy()
    adata.obs["counts_after_gene_filter"] = totals
    normalized = x.multiply((target_sum / totals)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    adata.X = normalized
    adata.uns["normalization"] = {
        "method": "total_count_then_log1p", "target_sum": float(target_sum),
        "qc_metrics": "computed on input genes, before gene filtering",
    }
    return adata


def mean_variance(x):
    """Column mean and unbiased variance, supporting sparse matrices."""
    x = x.astype(np.float64)
    mean = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.power(2).mean(axis=0)).ravel() if sparse.issparse(x) else (x**2).mean(axis=0)
    variance = np.maximum(second - mean**2, 0)
    if x.shape[0] > 1:
        variance *= x.shape[0] / (x.shape[0] - 1)
    return mean, variance


def dispersion_scores(x, n_bins=20):
    """Mean-bin standardized log(var/mean) on normalized, unlogged values.

    This transparent educational HVG recipe is dispersion based. It is not
    a claim of bitwise equivalence to Scanpy/Seurat's batch-aware selectors.
    """
    mean, variance = mean_variance(x)
    valid = (mean > 0) & (variance > 0)
    scores = np.full(x.shape[1], -np.inf)
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return scores
    log_mean = np.log1p(mean[valid])
    log_dispersion = np.log(variance[valid] / mean[valid])
    bins = pd.cut(log_mean, bins=n_bins, labels=False, duplicates="drop")
    for level in np.unique(bins):
        mask = bins == level
        values = log_dispersion[mask]
        if len(values) == 1:
            scores[ids[mask]] = 1.0
        else:
            std = values.std(ddof=1)
            scores[ids[mask]] = (values - values.mean()) / std if std > 0 else 0.0
    return scores


def select_highly_variable_genes(adata, source_key, n_top_genes=2000, n_bins=20):
    """Select HVGs per source, then combine by recurrence and median rank.

    First prefer genes selected in more sources; break ties with median rank
    among sources that selected that gene, then gene order. Mitochondrial
    genes are excluded from PCA features, but retained in X/counts.
    """
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be positive.")
    normalized = adata.X.copy()
    normalized.data = np.expm1(normalized.data)
    sources = adata.obs[source_key].to_numpy()
    levels = sorted(pd.unique(sources))
    ranks = np.full((len(levels), adata.n_vars), np.nan)
    allowed = ~adata.var["is_mito"].to_numpy()
    for i, source in enumerate(levels):
        scores = dispersion_scores(normalized[sources == source], n_bins)
        valid = np.flatnonzero(allowed & np.isfinite(scores))
        ordered = valid[np.argsort(-scores[valid], kind="stable")]
        selected = ordered[:n_top_genes]
        ranks[i, selected] = np.arange(len(selected))
    n_sources = np.sum(np.isfinite(ranks), axis=0)
    median_rank = np.full(adata.n_vars, np.inf)
    candidates = np.flatnonzero(n_sources > 0)
    if len(candidates) < 2:
        raise ValueError("Too few variable genes for PCA.")
    median_rank[candidates] = np.nanmedian(ranks[:, candidates], axis=0)
    order = np.lexsort((candidates, median_rank[candidates], -n_sources[candidates]))
    selected = candidates[order[:n_top_genes]]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[selected] = True
    adata.var["highly_variable"] = mask
    adata.var["hvg_n_sources"] = n_sources
    adata.var["hvg_median_rank"] = median_rank
    adata.uns["hvg"] = {
        "method": "per-source mean-bin log-dispersion, recurrence then median rank",
        "n_bins": n_bins, "n_top_genes": int(mask.sum()), "exclude_mito": True,
    }
    return adata


def calculate_pca(adata, n_components=30, clip_value=10.0, random_state=0):
    """Scale only HVGs, clip scaled values, and compute centered randomized PCA."""
    mask = adata.var["highly_variable"].to_numpy()
    x = adata.X[:, mask].toarray().astype(np.float64)
    means = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    x = (x - means) / std
    x = np.clip(x, -clip_value, clip_value)
    n_components = min(n_components, x.shape[0] - 1, x.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Too few cells or HVGs for at least two PCs.")
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    adata.obsm["X_pca"] = model.fit_transform(x).astype(np.float32)
    loadings = np.zeros((adata.n_vars, n_components), dtype=np.float32)
    loadings[mask] = model.components_.T
    adata.varm["PCs"] = loadings
    adata.var["scale_mean"] = np.nan
    adata.var["scale_std"] = np.nan
    adata.var.loc[mask, "scale_mean"] = means
    adata.var.loc[mask, "scale_std"] = std
    adata.uns["pca"] = {
        "variance_ratio": model.explained_variance_ratio_,
        "variance": model.explained_variance_, "clip_value": clip_value,
        "random_state": random_state, "center_after_clipping": model.mean_,
    }
    return adata
