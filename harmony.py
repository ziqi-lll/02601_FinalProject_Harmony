# Name: Ziqi Liu, Zhe Zhang
# Date: 2026.10.23
# Project Checkpoint 1: Harmony2 Implementation
# This file implements Harmony2 batch correction using NumPy and pandas.
# Algorithm reference: slowkow/harmonypy 2.1.0, src/harmony.cpp
# Commit: 19de40ddfb67d6733ea0ce969f0ede3e282198d8
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row to unit length, leaving zero rows unchanged.

    Args:
        matrix: A two-dimensional array to normalize.

    Returns:
        A float64 array with each nonzero row scaled to length one.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    normalized = np.zeros_like(matrix)

    # Normalize each nonzero row.
    for i in range(matrix.shape[0]):
        row = matrix[i]

        squared_values = row ** 2
        squared_sum = np.sum(squared_values)
        length = np.sqrt(squared_sum)

        if length > 0:
            normalized[i] = row / length

    return normalized


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert each row of scores into a probability distribution.

    Args:
        logits: A score matrix with cells as rows and clusters as columns.

    Returns:
        A probability matrix with the same shape, where each row sums to one.
    """
    shifted = logits.copy()

    # Shift each row for numerical stability.
    for i in range(logits.shape[0]):
        row_max = np.max(logits[i])
        shifted[i] -= row_max

    weights = np.exp(shifted)
    probabilities = np.zeros_like(weights)

    # Normalize each row to sum to one.
    for i in range(weights.shape[0]):
        row_sum = np.sum(weights[i])
        probabilities[i] = weights[i] / row_sum

    return probabilities


def encode_batch_column(labels: pd.Series,
                        key: str) -> tuple[np.ndarray, int, list[str]]:
    """
    Encode one metadata column as a one-hot matrix.

    Args:
        labels: A series containing one category label per cell.
        key: The metadata column name.

    Returns:
        A tuple containing the indicator matrix, category count,
        and category names.
    """
    if labels.isna().any():
        raise ValueError(f"Missing source labels in {key}.")

    codes, levels = pd.factorize(labels, sort=True)
    n_levels = len(levels)

    if n_levels == 0:
        raise ValueError("No cells were supplied.")

    indicators = np.zeros(
        (len(labels), n_levels),
        dtype=np.float64,
    )

    for i in range(len(labels)):
        category = codes[i]
        indicators[i, category] = 1.0

    names = []

    for level in levels:
        names.append(f"{key}={level}")

    return indicators, n_levels, names


def encode_batches(metadata: pd.DataFrame,
                   vars_use: str | list[str]
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert categorical metadata into a combined one-hot matrix.

    Args:
        metadata: Metadata containing one batch label per cell and variable.
        vars_use: A column name or list of distinct column names to encode.

    Returns:
        A tuple containing the cells-by-categories indicator matrix,
        category counts per variable, and category names.
    """
    if isinstance(vars_use, str):
        vars_use = [vars_use]

    if not vars_use:
        raise ValueError(
            "vars_use must contain distinct column names."
        )

    if len(set(vars_use)) != len(vars_use):
        raise ValueError(
            "vars_use must contain distinct column names."
        )

    parts = []
    sizes = []
    names = []

    for key in vars_use:
        labels = pd.Series(metadata[key])
        indicators, n_levels, column_names = encode_batch_column(
            labels, key
        )

        parts.append(indicators)
        sizes.append(n_levels)
        names.extend(column_names)

    phi = np.column_stack(parts)
    level_counts = np.asarray(sizes, dtype=int)

    return phi, level_counts, names


def expand_parameter(value: float | list[float] | np.ndarray,
                     level_counts: np.ndarray,
                     name: str) -> np.ndarray:
    """
    Expand parameter values into one value per batch category.

    Args:
        value: A scalar, one value per variable, or one value per category.
        level_counts: The number of categories in each variable.
        name: The parameter name used in error messages.

    Returns:
        A one-dimensional array of finite, nonnegative category values.
    """
    values = np.asarray(value, dtype=float)

    n_variables = len(level_counts)
    n_categories = int(np.sum(level_counts))

    # Use one shared value for all categories.
    if values.ndim == 0:
        result = np.full(n_categories, float(values))

    elif values.ndim == 1 and len(values) == n_variables:
        expanded = []

        # Repeat each variable's value for its categories.
        for i in range(n_variables):
            for _ in range(level_counts[i]):
                expanded.append(values[i])

        result = np.asarray(expanded, dtype=float)

    # Keep parameters already specified per category.
    elif values.ndim == 1 and len(values) == n_categories:
        result = values.copy()

    else:
        raise ValueError(f"Invalid length for {name}.")

    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(
            f"{name} must be finite and nonnegative."
        )

    return result


def sample_centroids(
    z_cos: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample distinct cell positions as initial cluster centers.

    Args:
        z_cos: A cells-by-dimensions matrix with normalized rows.
        n_clusters: The number of centers to sample.
        rng: A NumPy random generator for sampling.

    Returns:
        A clusters-by-dimensions matrix of sampled centers.
    """
    n_cells = len(z_cos)
    random_values = rng.random(n_clusters)
    indices = np.zeros(n_clusters, dtype=int)

    for k in range(n_clusters):
        index = int(np.floor(random_values[k] * n_cells + 5e-1))

        if index >= n_cells:
            index = n_cells - 1

        indices[k] = index

    centers = z_cos[indices].copy()
    chosen = np.zeros(n_cells, dtype=bool)

    for k in range(n_clusters):
        similarities = np.matmul(z_cos, centers[k])
        distance = np.abs(2.0 * (1.0 - similarities))

        draws = rng.random(n_cells)
        draws = np.maximum(draws, np.finfo(float).tiny)
        scores = -np.log(draws) / (distance + 1e-10)

        # Exclude cells already selected as centers.
        scores[chosen] = np.inf

        selected = int(np.argmin(scores))
        centers[k] = z_cos[selected]
        chosen[selected] = True

    return centers


def update_centroids(
    z_cos: np.ndarray,
    centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign cells to their nearest centers and calculate cluster means.

    Args:
        z_cos: A cells-by-dimensions matrix with normalized rows.
        centers: The current clusters-by-dimensions center matrix.

    Returns:
        A tuple containing updated centers, cell counts per cluster,
        and the assigned cluster index for each cell.
    """
    n_cells = len(z_cos)
    n_clusters = len(centers)

    cell_squares = np.sum(z_cos ** 2, axis=1)
    center_squares = np.sum(centers ** 2, axis=1)
    dot_products = np.matmul(z_cos, centers.T)
    distances = np.zeros_like(dot_products)

    for k in range(n_clusters):
        distances[:, k] = (
            cell_squares + center_squares[k]
            - 2 * dot_products[:, k]
        )

    labels = np.argmin(distances, axis=1)
    counts = np.zeros(n_clusters, dtype=int)
    updated = np.zeros_like(centers)

    for i in range(n_cells):
        cluster = labels[i]
        counts[cluster] += 1
        updated[cluster] += z_cos[i]

    for k in range(n_clusters):
        if counts[k] > 0:
            updated[k] /= counts[k]

    return updated, counts, labels


def recover_empty_centroids(
    z_cos: np.ndarray,
    updated: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Replace empty cluster centers with selected cell positions.

    Args:
        z_cos: A cells-by-dimensions matrix with normalized rows.
        updated: The center matrix to modify in place.
        counts: The number of cells assigned to each cluster.
        labels: The assigned cluster index for each cell.
        rng: A NumPy random generator for fallback sampling.

    Returns:
        The updated center matrix with empty centers replaced.
    """
    n_cells = len(z_cos)
    n_clusters = len(updated)
    donors = []

    for k in range(n_clusters - 1, -1, -1):
        if counts[k] >= 2:
            donors.append(k)

    empty_clusters = np.flatnonzero(counts == 0)

    for j, k in enumerate(empty_clusters):
        if j < len(donors):
            donor = donors[j]

            # Use the donor cluster's last cell in the original row order.
            donor_cells = np.flatnonzero(labels == donor)
            index = int(donor_cells[-1])
        else:
            index = rng.integers(n_cells)

        updated[k] = z_cos[index]

    return updated


def initialize_centroids(
    z_cos: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Initialize centers using weighted sampling and 10 clustering rounds.

    Args:
        z_cos: A cells-by-dimensions matrix with normalized rows.
        n_clusters: The number of cluster centers to initialize.
        rng: A NumPy random generator for sampling cell positions.

    Returns:
        A clusters-by-dimensions matrix of normalized cluster centers.
    """
    centers = sample_centroids(z_cos, n_clusters, rng)

    for _ in range(10):
        updated, counts, labels = update_centroids(z_cos, centers)

        centers = recover_empty_centroids(
            z_cos,
            updated,
            counts,
            labels,
            rng
        )

    return normalize_rows(centers)


def assignment_probabilities(distances: np.ndarray,
                             sigma: np.ndarray,
                             expected: np.ndarray,
                             observed: np.ndarray,
                             theta: np.ndarray,
                             phi: np.ndarray) -> np.ndarray:
    """
    Calculate soft cluster assignments using distance and batch diversity.

    Args:
        distances: A cells-by-clusters matrix of distances.
        sigma: A positive distance scale for each cluster.
        expected: A clusters-by-categories matrix of expected counts.
        observed: A clusters-by-categories matrix of observed soft counts.
        theta: The diversity penalty strength for each category.
        phi: A cells-by-categories indicator matrix.

    Returns:
        A cells-by-clusters probability matrix with each row summing to one.
    """
    safe_expected = np.maximum(expected, 0)
    safe_total = np.maximum(observed + expected, 0)

    log_expected = np.log1p(2 * safe_expected)
    log_total = np.log1p(safe_total)
    log_ratio = log_expected - log_total

    diversity_columns = []

    # Weight diversity scores by batch category.
    for b in range(len(theta)):
        column = log_ratio[:, b] * theta[b]
        diversity_columns.append(column)

    log_diversity = np.column_stack(diversity_columns)

    distance_columns = []

    # Scale distance scores by cluster.
    for k in range(len(sigma)):
        column = -distances[:, k] / sigma[k]
        distance_columns.append(column)

    distance_scores = np.column_stack(distance_columns)

    # Combine distance and category scores before softmax.
    diversity_scores = np.matmul(phi, log_diversity.T)
    logits = distance_scores + diversity_scores

    return softmax(logits)


def update_assignments(r: np.ndarray,
                       distances: np.ndarray,
                       phi: np.ndarray,
                       proportions: np.ndarray,
                       sigma: np.ndarray,
                       theta: np.ndarray,
                       block_size: float,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update soft cluster assignments in randomly ordered blocks of cells.

    Args:
        r: A cells-by-clusters matrix of current assignment probabilities.
        distances: A cells-by-clusters matrix of distances.
        phi: A cells-by-categories indicator matrix.
        proportions: The fraction of cells in each batch category.
        sigma: A positive distance scale for each cluster.
        theta: The diversity penalty strength for each category.
        block_size: The fraction of cells to update in each block.
        rng: A NumPy random generator for shuffling cell indices.

    Returns:
        A tuple containing updated assignment probabilities,
        observed category counts, and expected category counts.
    """
    r = r.copy()
    n_cells = len(r)

    # Calculate the current category totals.
    observed = np.matmul(r.T, phi)
    cluster_totals = np.sum(r, axis=0)
    expected = np.outer(cluster_totals, proportions)

    # Shuffle cells and define the update blocks.
    order = rng.permutation(n_cells)
    n_blocks = int(np.ceil(1.0 / block_size))

    block_length = int(n_cells * block_size)

    if block_length < 1:
        block_length = 1

    # Update one shuffled cell block at a time.
    for block in range(n_blocks):
        start = block * block_length

        if start >= n_cells:
            break

        if block == n_blocks - 1:
            stop = n_cells
        else:
            stop = start + block_length

            if stop > n_cells:
                stop = n_cells

        ids = order[start:stop]
        old = r[ids]
        block_phi = phi[ids]

        # Remove the block's old contribution.
        old_observed = np.matmul(old.T, block_phi)
        old_totals = np.sum(old, axis=0)
        old_expected = np.outer(old_totals, proportions)

        observed -= old_observed
        expected -= old_expected

        # Reassign the block using the remaining statistics.
        new = assignment_probabilities(
            distances[ids],
            sigma,
            expected,
            observed,
            theta,
            block_phi,
        )

        r[ids] = new

        # Add the block's new contribution.
        new_observed = np.matmul(new.T, block_phi)
        new_totals = np.sum(new, axis=0)
        new_expected = np.outer(new_totals, proportions)

        observed += new_observed
        expected += new_expected

    # Calculate the current category totals.
    observed = np.matmul(r.T, phi)
    cluster_totals = np.sum(r, axis=0)
    expected = np.outer(cluster_totals, proportions)

    return r, observed, expected


def calculate_objective(r: np.ndarray,
                        distances: np.ndarray,
                        expected: np.ndarray,
                        observed: np.ndarray,
                        sigma: np.ndarray,
                        theta: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Calculate the clustering objective from distance, entropy, and diversity.

    Args:
        r: A cells-by-clusters matrix of assignment probabilities.
        distances: A cells-by-clusters matrix of distances.
        expected: A clusters-by-categories matrix of expected counts.
        observed: A clusters-by-categories matrix of observed soft counts.
        sigma: The entropy and diversity weight for each cluster.
        theta: The diversity penalty strength for each category.

    Returns:
        A tuple containing the total objective and an array of its distance,
        entropy, and diversity terms, all scaled by 2000 / number of cells.
    """
    # Calculate probability-weighted distance.
    distance_term = np.sum(r * distances)

    safe_r = np.maximum(r, np.finfo(float).tiny)
    log_r = np.log(safe_r)

    entropy_columns = []

    # Calculate cluster-weighted entropy contributions.
    for k in range(len(sigma)):
        column = r[:, k] * log_r[:, k]
        column *= sigma[k]
        entropy_columns.append(column)

    entropy_values = np.column_stack(entropy_columns)
    entropy_term = np.sum(entropy_values)

    log_total = np.log1p(observed + expected)
    log_expected = np.log1p(2 * expected)
    log_ratio = log_total - log_expected

    diversity_columns = []

    # Calculate weighted batch-diversity contributions.
    for b in range(len(theta)):
        column = observed[:, b] * log_ratio[:, b]
        column *= theta[b]
        column *= sigma
        diversity_columns.append(column)

    diversity_values = np.column_stack(diversity_columns)
    diversity_term = np.sum(diversity_values)

    # Combine and scale the three objective terms.
    terms = np.array([
        distance_term,
        entropy_term,
        diversity_term,
    ])

    scale = 2000 / len(r)
    terms *= scale

    total = float(np.sum(terms))

    return total, terms


def objective_converged(old: float, new: float, tolerance: float) -> bool:
    """
    Check whether the objective has decreased by less than the tolerance.

    Args:
        old: The previous objective value.
        new: The current objective value.
        tolerance: The threshold for the relative objective decrease.

    Returns:
        True if the relative decrease is nonnegative and below tolerance,
        or both objective values are zero. Otherwise, False.
    """
    # Reject invalid objective values or tolerance.
    if not np.isfinite(old):
        return False

    if not np.isfinite(new):
        return False

    if not np.isfinite(tolerance):
        return False

    # Handle a zero denominator.
    if old == 0:
        if new == 0:
            return True
        else:
            return False

    difference = old - new
    relative_change = difference / abs(old)

    # An increase does not count as convergence.
    if relative_change < 0:
        return False

    if relative_change < tolerance:
        return True

    return False


def select_batch_levels(observed: np.ndarray,
                        batch_sizes: np.ndarray,
                        level_counts: np.ndarray,
                        cutoff: float) -> np.ndarray:
    """
    Select batch categories eligible for correction in one cluster.

    Args:
        observed: Observed soft counts for this cluster.
        batch_sizes: The total number of cells in each category.
        level_counts: The number of categories in each variable.
        cutoff: The strict minimum observed-to-batch-size ratio.

    Returns:
        A boolean mask retaining at least two categories per variable.
        Variables with fewer qualifying categories are excluded.
    """
    keep = observed / batch_sizes > cutoff
    first = 0

    for count in level_counts:
        last = first + count

        if np.sum(keep[first:last]) < 2:
            keep[first:last] = False

        first = last

    return keep


def build_ridge_penalties(expected: np.ndarray,
                          keep: np.ndarray,
                          alpha: float,
                          lamb: np.ndarray | None) -> np.ndarray:
    """
    Build intercept and category penalties for one cluster.

    Args:
        expected: Expected category counts for this cluster.
        keep: A boolean mask of retained categories.
        alpha: The multiplier for adaptive penalties.
        lamb: Fixed intercept and category penalties, or None.

    Returns:
        A penalty array with the intercept first, followed by
        retained category penalties.
    """
    n_kept = int(np.sum(keep))
    penalties = np.zeros(n_kept + 1)

    if lamb is None:
        penalties[1:] = alpha * expected[keep]
        return penalties

    # Apply the fixed intercept penalty only when all levels are kept.
    if np.all(keep):
        penalties[0] = lamb[0]

    batch_penalties = lamb[1:]
    penalties[1:] = batch_penalties[keep]

    return penalties


def fit_ridge_model(z_orig: np.ndarray,
                    batch_design: np.ndarray,
                    weights: np.ndarray,
                    penalties: np.ndarray) -> np.ndarray:
    """
    Fit a weighted ridge model to the original coordinates.

    Args:
        z_orig: The original cells-by-PCs coordinate matrix.
        batch_design: The indicator matrix for retained categories.
        weights: The assignment weight of each cell for this cluster.
        penalties: Intercept and category ridge penalties.

    Returns:
        A coefficient matrix with the intercept in the first row
        and category effects in the remaining rows.
    """
    eligible = np.any(batch_design > 0, axis=1)
    intercept = eligible.astype(float)
    design = np.column_stack([intercept, batch_design])
    weighted = design.copy()

    for j in range(design.shape[1]):
        weighted[:, j] *= weights

    gram = np.matmul(design.T, weighted)

    for j in range(len(penalties)):
        gram[j, j] += penalties[j]

    rhs = np.matmul(weighted.T, z_orig)

    try:
        cholesky = np.linalg.cholesky(gram)
        intermediate = np.linalg.solve(cholesky, rhs)
        coefficients = np.linalg.solve(
            cholesky.T, intermediate
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Singular ridge model; use positive lamb or alpha."
        ) from error

    return coefficients


def ridge_correction(
    z_orig: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
    level_counts: np.ndarray,
    expected: np.ndarray,
    observed: np.ndarray,
    centers: np.ndarray,
    alpha: float = 2e-1,
    lamb: np.ndarray | None = None,
    batch_prop_cutoff: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Remove batch effects using a weighted ridge model for each cluster.

    Args:
        z_orig: The original cells-by-PCs coordinate matrix.
        r: A cells-by-clusters matrix of assignment probabilities.
        phi: A cells-by-categories indicator matrix.
        level_counts: The number of categories in each variable.
        expected: A clusters-by-categories matrix of expected counts.
        observed: A clusters-by-categories matrix of observed soft counts.
        centers: The current clusters-by-PCs center matrix.
        alpha: The multiplier for adaptive ridge penalties.
        lamb: Intercept and category penalties, or None for adaptive ridge.
        batch_prop_cutoff: The strict category-retention threshold.

    Returns:
        A tuple containing corrected coordinates, normalized centers,
        and the number of retained categories per cluster.
    """
    z_corr = z_orig.copy()
    centers = centers.copy()
    batch_sizes = np.sum(phi, axis=0)
    kept_per_cluster = []

    for k in range(r.shape[1]):
        keep = select_batch_levels(
            observed[k],
            batch_sizes,
            level_counts,
            batch_prop_cutoff
        )

        n_kept = int(np.sum(keep))
        kept_per_cluster.append(n_kept)

        if n_kept == 0:
            continue

        batch_design = phi[:, keep]
        weights = r[:, k]

        penalties = build_ridge_penalties(
            expected[k],
            keep,
            alpha,
            lamb
        )

        coefficients = fit_ridge_model(
            z_orig,
            batch_design,
            weights,
            penalties
        )

        centers[k] = coefficients[0]
        batch_effect = np.matmul(
            batch_design, coefficients[1:]
        )

        for pc in range(z_orig.shape[1]):
            correction = batch_effect[:, pc] * weights
            z_corr[:, pc] -= correction

    return z_corr, normalize_rows(centers), kept_per_cluster


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use: str | list[str],
    theta: float | list[float] | np.ndarray | None = 2.0,
    lamb: float | list[float] | np.ndarray | None = None,
    sigma: float | list[float] | np.ndarray = 0.1,
    nclust: int | None = None,
    tau: float = 0.0,
    block_size: float = 0.05,
    max_iter_harmony: int = 10,
    max_iter_kmeans: int = 4,
    epsilon_cluster: float = 1e-3,
    epsilon_harmony: float = 1e-2,
    alpha: float = 0.2,
    batch_prop_cutoff: float = 1e-5,
    random_state: int | None = 0,
    verbose: bool = True
) -> dict:
    """
    Run Harmony2 batch correction on a cells-by-PCs coordinate matrix.

    Args:
        data_mat: A finite cells-by-PCs matrix.
        meta_data: Metadata with cells in the same row order as data_mat.
        vars_use: A column name or list of categorical columns to correct.
        theta: Diversity strengths specified as a scalar, per variable,
            or per category. Defaults to 2.0.
        lamb: Fixed ridge penalties specified as a scalar, per variable,
            per category, or a full vector including the intercept.
            None or -1 selects adaptive penalties.
        sigma: A positive distance scale, shared or specified per cluster.
        nclust: The number of clusters, or None to choose from the cell count.
        tau: A parameter that reduces diversity pressure for small categories.
            Zero disables this adjustment.
        block_size: The fraction of cells updated in each assignment block.
        max_iter_harmony: The maximum number of outer correction iterations.
        max_iter_kmeans: The maximum assignment updates per outer iteration.
        epsilon_cluster: The tolerance for inner-loop convergence.
        epsilon_harmony: The tolerance for outer-loop convergence.
        alpha: The multiplier for adaptive ridge penalties.
        batch_prop_cutoff: The category-retention threshold for ridge
                           correction.
        random_state: The seed for NumPy's random generator.
        verbose: Whether to print iteration progress.

    Returns:
        A dictionary containing corrected coordinates Z_corr (cells x PCs),
        assignment probabilities R (cells x clusters), cluster centers Y
        (PCs x clusters), objective histories, convergence information,
        batch names, and retained-category counts.
        If max_iter_harmony is zero, Z_corr contains normalized input
        coordinates.
    """
    # Validate data and parameters.
    z_orig = np.asarray(data_mat, dtype=np.float64).copy()
    if (
        z_orig.ndim != 2
        or min(z_orig.shape) == 0
        or not np.isfinite(z_orig).all()
    ):
        raise ValueError(
            "data_mat must be a nonempty finite cells x PCs matrix."
            )
    phi, level_counts, batch_names = encode_batches(meta_data, vars_use)
    n_cells = len(z_orig)
    if len(phi) != n_cells:
        raise ValueError(
            "Metadata and embedding must have the same cell count."
            )

    if nclust is None:
        nclust = max(1, min(round(n_cells / 30), 100))

    if not isinstance(nclust, (int, np.integer)) or not 1 <= nclust <= n_cells:
        raise ValueError("nclust must be an integer between 1 and n_cells.")

    # Check the iteration limits.
    for value, name in [(max_iter_harmony, "max_iter_harmony"),
                        (max_iter_kmeans, "max_iter_kmeans")]:

        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")

    if not np.isfinite(block_size) or not 0 < block_size <= 1:
        raise ValueError("block_size must be in (0, 1].")

    # Check nonnegative strengths and thresholds.
    for value, name in [(alpha, "alpha"), (tau, "tau"),
                        (epsilon_cluster, "epsilon_cluster"),
                        (epsilon_harmony, "epsilon_harmony"),
                        (batch_prop_cutoff, "batch_prop_cutoff")]:

        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")

    # Prepare per-cluster and per-category parameters.
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.full(nclust, float(sigma)) if sigma.ndim == 0 else sigma

    if (
        sigma.shape != (nclust,)
        or not np.isfinite(sigma).all()
        or (sigma <= 0).any()
    ):
        raise ValueError(
            "sigma must be positive, scalar or one per cluster."
            )

    theta = expand_parameter(2.0 if theta is None else theta,
                             level_counts,
                             "theta")
    sizes = phi.sum(axis=0)
    proportions = sizes / n_cells
    # Reduce diversity pressure for small categories.
    if tau > 0:
        theta *= 1 - np.exp(-(sizes / (nclust * tau)) ** 2)
    # Prepare fixed penalties or select adaptive ridge.

    if lamb is not None:
        values = np.asarray(lamb, dtype=float)

        if values.size == 1 and values.item() == -1:
            lamb = None

        elif values.ndim == 1 and len(values) == phi.shape[1] + 1:
            lamb = values.copy()

        else:
            lamb = np.r_[0.0, expand_parameter(lamb, level_counts, "lamb")]

        if lamb is not None and (
            not np.isfinite(lamb).all()
            or (lamb < 0).any()
        ):
            raise ValueError(
                "lamb must be finite and nonnegative."
                )

    # Initialize centers, probabilities, and objective histories.
    rng = np.random.default_rng(random_state)
    z_corr = normalize_rows(z_orig)
    centers = initialize_centroids(z_corr, nclust, rng)
    distances = 2 * (1 - z_corr @ centers.T)
    r = softmax(-distances / sigma[None, :])
    observed = r.T @ phi
    expected = np.outer(r.sum(axis=0), proportions)
    initial, terms = calculate_objective(r,
                                         distances,
                                         expected,
                                         observed,
                                         sigma,
                                         theta)
    objectives = [initial]
    inner_objectives, objective_terms = [initial], [terms]
    inner_rounds, kept_history = [], []
    converged = False

    # Alternate assignment updates and ridge correction.
    for iteration in range(max_iter_harmony):
        # Rebuild assignments from the latest corrected coordinates.
        if iteration > 0:
            z_cos = normalize_rows(z_corr)
            distances = 2 * (1 - z_cos @ centers.T)
            r = softmax(-distances / sigma[None, :])
            observed = r.T @ phi
            expected = np.outer(r.sum(axis=0), proportions)
        rounds = 0
        # Refine soft cluster assignments.
        for inner in range(max_iter_kmeans):
            r, observed, expected = update_assignments(
                r, distances, phi, proportions, sigma, theta, block_size, rng
            )
            objective, terms = calculate_objective(
                r, distances, expected, observed, sigma, theta
            )
            inner_objectives.append(objective)
            objective_terms.append(terms)
            rounds = inner + 1
            # Check the inner objective window for convergence.
            if inner > 3:
                old = sum(inner_objectives[-4:-1])
                new = sum(inner_objectives[-3:])
                if old != 0 and abs(old - new) / abs(old) < epsilon_cluster:
                    break
        inner_rounds.append(rounds)
        objectives.append(inner_objectives[-1])
        # Correct original PCs using the latest assignments.
        z_corr, centers, kept = ridge_correction(
            z_orig, r, phi, level_counts, expected, observed, centers,
            alpha, lamb, batch_prop_cutoff,
        )
        kept_history.append(kept)
        if not np.isfinite(z_corr).all() or not np.isfinite(r).all():
            raise FloatingPointError("Harmony produced non-finite values.")
        if verbose:
            print(f"  Python Harmony2 {iteration + 1}/{max_iter_harmony}: "
                  f"objective={objectives[-1]:.6f}", flush=True)
        # Check outer convergence.
        converged = objective_converged(objectives[-2],
                                        objectives[-1],
                                        epsilon_harmony)
        if converged:
            break

    return {
        "Z_corr": z_corr,
        "R": r,
        "Y": centers.T,
        "objective_harmony": objectives,
        "objective_kmeans": inner_objectives,
        "objective_terms": np.asarray(objective_terms),
        "kmeans_rounds": inner_rounds,
        "converged": bool(converged),
        "n_iter": len(inner_rounds),
        "batch_names": batch_names,
        "kept_batch_levels": kept_history,
    }
