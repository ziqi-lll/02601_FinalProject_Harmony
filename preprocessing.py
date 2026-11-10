"""Preprocessing for the Harmony2 project.

Steps:
1. Load the original count matrix.
2. Calculate quality-control (QC) statistics.
3. Filter cells and genes.
4. Normalize counts and apply log1p.
5. Select highly variable genes (HVGs) from all sources.
6. Run PCA on the selected genes.

The expression matrix remains sparse until PCA.  Input counts must be
nonnegative and unlogged; fractional values are allowed.

This version keeps the original calculations, settings, and output fields,
but uses more explicit steps and beginner-friendly comments.
"""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA


def load_input(file_path, source_key):
    """Load an h5ad file and return the data and its source labels."""
    adata = ad.read_h5ad(file_path)

    # Make sure the input contains cells and genes.
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("The input expression matrix is empty.")

    # Each cell and each gene must have a unique ID.
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell IDs and gene IDs must be unique.")

    # The specified source column must exist and have no missing values.
    if source_key not in adata.obs:
        raise ValueError(f"Source column not found: {source_key}")

    source_labels = adata.obs[source_key].copy()
    if source_labels.isna().any():
        raise ValueError("Some cells have missing source labels.")

    # Check the expression values without converting a large sparse matrix
    # into a dense matrix. Sparse matrices store their nonzero entries in .data.
    if sparse.issparse(adata.X):
        values = adata.X.data
    else:
        values = np.asarray(adata.X)

    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("X must contain finite, nonnegative count-like values.")

    # Use the same sparse matrix type and precision as the original code.
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    adata.X.sum_duplicates()
    adata.X.eliminate_zeros()

    return adata, source_labels


def calculate_qc_metrics(adata, mito_gene_ids):
    """Calculate cell QC using the original, unfiltered counts.

    mito_gene_ids contains the 37 human mitochondrial Ensembl gene IDs.
    Ensembl version suffixes (such as '.5') are ignored.
    The MT- gene-symbol shortcut is not complete for LV_raw.h5ad.
    """
    x = adata.X

    # Total expression counts for each cell.
    total_counts = np.asarray(x.sum(axis=1)).ravel()

    # Number of genes expressed in each cell (expression > 0).
    detected = x > 0
    n_genes = np.asarray(detected.sum(axis=1)).ravel()

    # Decide which genes belong to the supplied mitochondrial gene list.
    is_mito = []
    for gene_id in adata.var_names:
        gene_without_version = gene_id.split(".")[0]
        is_mito.append(gene_without_version in mito_gene_ids)
    is_mito = np.array(is_mito, dtype=bool)

    if not is_mito.any():
        raise ValueError("No mitochondrial IDs matched. Check gene identifiers.")

    # Count mitochondrial expression in every cell.
    mito_matrix = x[:, is_mito]
    mito_counts = np.asarray(mito_matrix.sum(axis=1)).ravel()

    # Cells without counts receive NaN instead of dividing by zero.
    pct_mito = np.full(adata.n_obs, np.nan)
    for i in range(adata.n_obs):
        if total_counts[i] > 0:
            pct_mito[i] = mito_counts[i] / total_counts[i] * 100

    # Store QC annotations; do not change the expression matrix.
    adata.var["is_mito"] = is_mito
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes"] = n_genes
    adata.obs["pct_mito"] = pct_mito

    return adata


def filter_cells(adata, min_genes=200, max_mito=20):
    """Keep cells with enough genes and sufficiently low mitochondrial % ."""
    enough_counts = adata.obs["total_counts"] > 0
    enough_genes = adata.obs["n_genes"] >= min_genes
    low_mito = adata.obs["pct_mito"] < max_mito

    # A cell must pass all three conditions.
    keep = enough_counts & enough_genes & low_mito
    filtered = adata[keep, :].copy()

    if filtered.n_obs == 0:
        raise ValueError("No cells remain. Check the QC thresholds.")

    return filtered


def filter_genes(adata, min_cells=3):
    """Keep genes expressed in at least min_cells retained cells."""
    if not isinstance(min_cells, (int, np.integer)) or min_cells < 1:
        raise ValueError("min_cells must be a positive integer.")

    # Count how many cells express each gene.
    detected = adata.X > 0
    n_cells = np.asarray(detected.sum(axis=0)).ravel()

    keep = n_cells >= min_cells
    if not keep.any():
        raise ValueError("No genes remain. Check min_cells.")

    filtered = adata[:, keep].copy()
    filtered.var["n_cells"] = n_cells[keep]

    return filtered


def normalize_log1p(adata, target_sum=1e4):
    """Normalize each cell's retained-gene counts, then apply log1p.

    adata.layers['counts'] saves the counts before normalization.
    QC values in adata.obs still describe the original (pre-filter) counts.
    Do not call this function twice on the same data.
    """
    if "counts" in adata.layers:
        raise ValueError("counts layer already exists; normalization may already be done.")

    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be positive and finite.")

    x = sparse.csr_matrix(adata.X, dtype=np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel()

    if np.any(totals <= 0):
        raise ValueError("A cell has zero counts after gene filtering.")

    # Save the original retained-gene counts for later use.
    adata.layers["counts"] = x.copy()
    adata.obs["counts_after_gene_filter"] = totals

    # Make each cell have approximately target_sum total counts.
    scale_factors = target_sum / totals
    normalized = x.multiply(scale_factors[:, None]).tocsr()

    # Apply log(1 + count) only to stored nonzero entries.
    normalized.data = np.log1p(normalized.data)
    adata.X = normalized

    adata.uns["normalization"] = {
        "method": "total_count_then_log1p",
        "target_sum": float(target_sum),
        "qc_metrics": "computed on input genes, before gene filtering",
    }

    return adata


def mean_variance(x):
    """Return each gene's mean and sample variance (ddof=1)."""
    x = x.astype(np.float64)
    n_cells = x.shape[0]

    # First calculate the average expression for each gene.
    mean = np.asarray(x.mean(axis=0)).ravel()

    # Calculate E[X^2] while keeping sparse data sparse.
    if sparse.issparse(x):
        squared = x.power(2)
    else:
        squared = x ** 2

    second_moment = np.asarray(squared.mean(axis=0)).ravel()

    # Variance = E[X^2] - E[X]^2. Avoid tiny negative rounding errors.
    variance = second_moment - mean ** 2
    variance = np.maximum(variance, 0)

    # Convert population variance into unbiased sample variance.
    if n_cells > 1:
        correction = n_cells / (n_cells - 1)
        variance = variance * correction

    return mean, variance


def dispersion_scores(x, n_bins=20):
    """Score genes using standardized log(variance / mean) within mean bins.

    Input x contains normalized counts BEFORE log1p, not log-transformed data.
    This is the original project's educational dispersion-based HVG method,
    not an exact reimplementation of Scanpy or Seurat HVG selection.
    """
    mean, variance = mean_variance(x)

    # Genes with zero mean or zero variance cannot receive a useful score.
    valid = (mean > 0) & (variance > 0)
    scores = np.full(x.shape[1], -np.inf)
    valid_indices = np.flatnonzero(valid)

    if len(valid_indices) == 0:
        return scores

    # Compute the mean and dispersion for genes that passed the check.
    valid_means = mean[valid_indices]
    valid_variances = variance[valid_indices]
    log_mean = np.log1p(valid_means)
    log_dispersion = np.log(valid_variances / valid_means)

    # Put genes with similar mean expression into the same bin.
    bins = pd.cut(log_mean, bins=n_bins, labels=False, duplicates="drop")

    for bin_number in np.unique(bins):
        in_this_bin = bins == bin_number
        gene_indices = valid_indices[in_this_bin]
        values = log_dispersion[in_this_bin]

        if len(values) == 1:
            # A single gene in a bin gets score 1, as in the original code.
            scores[gene_indices] = 1.0
        else:
            bin_mean = values.mean()
            bin_std = values.std(ddof=1)
            if bin_std > 0:
                scores[gene_indices] = (values - bin_mean) / bin_std
            else:
                scores[gene_indices] = 0.0

    return scores


def select_highly_variable_genes(adata, source_key, n_top_genes=2000, n_bins=20):
    """Choose HVGs for each source, then combine the source-level selections.

    Priority 1: a gene is selected in more sources.
    Priority 2: the gene has a better (smaller) median selected rank.
    Priority 3: the gene appears earlier in the original gene list.
    Mitochondrial genes stay in X and counts but are excluded from the HVGs.
    """
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be positive.")

    # HVG scoring uses normalized counts, not log1p values.
    # Reverse log1p on a COPY; adata.X remains unchanged.
    normalized = adata.X.copy()
    normalized.data = np.expm1(normalized.data)

    sources = adata.obs[source_key].to_numpy()
    levels = sorted(pd.unique(sources))
    n_genes = adata.n_vars

    # Each row represents a source and each column represents a gene.
    # NaN means this gene was not selected in this source.
    ranks = np.full((len(levels), n_genes), np.nan)
    allowed = ~adata.var["is_mito"].to_numpy()

    for source_index in range(len(levels)):
        source = levels[source_index]
        source_cells = sources == source
        source_matrix = normalized[source_cells]
        scores = dispersion_scores(source_matrix, n_bins)

        # Find non-mitochondrial genes with valid dispersion scores.
        valid_indices = np.flatnonzero(allowed & np.isfinite(scores))

        # Highest score first. Stable sorting preserves original gene order
        # when scores are exactly tied.
        sort_positions = np.argsort(-scores[valid_indices], kind="stable")
        sorted_genes = valid_indices[sort_positions]
        selected_genes = sorted_genes[:n_top_genes]

        # Record each selected gene's rank (0 = best in this source).
        for rank in range(len(selected_genes)):
            gene_index = selected_genes[rank]
            ranks[source_index, gene_index] = rank

    # Count the number of sources that selected each gene.
    n_sources = np.sum(np.isfinite(ranks), axis=0)

    # Find all genes selected at least once.
    candidates = np.flatnonzero(n_sources > 0)
    if len(candidates) < 2:
        raise ValueError("Too few variable genes for PCA.")

    # Calculate the median rank using only sources that selected the gene.
    median_rank = np.full(n_genes, np.inf)
    for gene_index in candidates:
        selected_ranks = []
        for source_index in range(len(levels)):
            rank = ranks[source_index, gene_index]
            if np.isfinite(rank):
                selected_ranks.append(rank)
        median_rank[gene_index] = np.median(selected_ranks)

    # Build an easy-to-read sorting rule instead of using np.lexsort().
    def gene_sort_key(gene_index):
        return (-n_sources[gene_index], median_rank[gene_index], gene_index)

    ordered_genes = sorted(candidates, key=gene_sort_key)
    selected = ordered_genes[:n_top_genes]

    # Mark the selected genes, without deleting the other genes.
    mask = np.zeros(n_genes, dtype=bool)
    for gene_index in selected:
        mask[gene_index] = True

    adata.var["highly_variable"] = mask
    adata.var["hvg_n_sources"] = n_sources
    adata.var["hvg_median_rank"] = median_rank
    adata.uns["hvg"] = {
        "method": "per-source mean-bin log-dispersion, recurrence then median rank",
        "n_bins": n_bins,
        "n_top_genes": int(mask.sum()),
        "exclude_mito": True,
    }

    return adata


def calculate_pca(adata, n_components=30, clip_value=10.0, random_state=0):
    """Scale and clip HVGs, then calculate PCA on the selected genes."""
    mask = adata.var["highly_variable"].to_numpy()

    # Only HVGs are used for PCA. This is the step that becomes dense.
    x = adata.X[:, mask].toarray().astype(np.float64)

    # Calculate the mean and sample standard deviation for each HVG.
    means = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    std[std == 0] = 1.0

    # Scale each gene: (expression - mean) / standard deviation.
    x = (x - means) / std

    # Limit extreme scaled values, just like the original code.
    x = np.clip(x, -clip_value, clip_value)

    # PCA cannot use more components than the sample or gene dimensions allow.
    n_components = min(n_components, x.shape[0] - 1, x.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Too few cells or HVGs for at least two PCs.")

    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    pcs = model.fit_transform(x)
    adata.obsm["X_pca"] = pcs.astype(np.float32)

    # Store PCA loadings for all genes. Non-HVG genes have zero loadings.
    loadings = np.zeros((adata.n_vars, n_components), dtype=np.float32)
    loadings[mask] = model.components_.T
    adata.varm["PCs"] = loadings

    # Save the means and standard deviations used for HVGs.
    adata.var["scale_mean"] = np.nan
    adata.var["scale_std"] = np.nan
    adata.var.loc[mask, "scale_mean"] = means
    adata.var.loc[mask, "scale_std"] = std

    adata.uns["pca"] = {
        "variance_ratio": model.explained_variance_ratio_,
        "variance": model.explained_variance_,
        "clip_value": clip_value,
        "random_state": random_state,
        "center_after_clipping": model.mean_,
    }

    return adata
