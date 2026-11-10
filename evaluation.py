"""Evaluate our Harmony2 implementation on liver and simulated data.

We check three things:
1. Do cells from different sources mix better after correction?
2. Are our corrected coordinates similar to official harmonypy?
3. Can Harmony remove a known batch shift without losing cell types?

The real liver data do not have verified cell-type labels. Source and donor
are confounded, so source mixing alone cannot prove technical batch removal.
"""

from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from harmony import run_harmony


METHODS = ["Uncorrected", "Python Harmony2", "harmonypy 2.1.0"]
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]


def qc_summary(before_obs, after_obs, source_key):
    """Count how many cells from each source survive preprocessing."""
    before = before_obs[source_key].value_counts().sort_index()
    after = after_obs[source_key].value_counts().reindex(before.index, fill_value=0)

    table = pd.DataFrame({"before": before, "after": after})
    table["removed"] = table["before"] - table["after"]
    table["retained_percent"] = 100 * table["after"] / table["before"]
    return table


def neighbor_indices(embedding, n_neighbors=30):
    """For each cell, find the indices of its closest OTHER cells."""
    n_cells = len(embedding)
    k = min(n_neighbors, n_cells - 1)
    if k < 1:
        raise ValueError("At least two cells are needed.")

    model = NearestNeighbors(n_neighbors=k + 1, n_jobs=1)
    model.fit(embedding)
    all_neighbors = model.kneighbors(embedding, return_distance=False)

    result = []
    for cell_index, row in enumerate(all_neighbors):
        others = row[row != cell_index][:k]
        result.append(others)
    return np.array(result)


def mixing_metrics(embedding, labels, n_neighbors=30):
    """Measure source mixing among the 30 nearest cells.

    The inverse Simpson calculation here uses a fixed number of neighbors.
    It is NOT the standard perplexity-based iLISI estimator from Harmony papers.
    """
    codes, sources = pd.factorize(labels, sort=True)
    neighbors = neighbor_indices(embedding, n_neighbors)

    # All neighbors from the same source = 1; all different = 0.
    same_source = codes[neighbors] == codes[:, None]
    same_fraction = float(same_source.mean())

    # Count the different sources among each cell's neighbors.
    inverse_simpson_scores = []
    for cell_neighbors in neighbors:
        neighbor_sources = codes[cell_neighbors]
        counts = np.bincount(neighbor_sources, minlength=len(sources))
        proportions = counts / len(cell_neighbors)
        score = 1 / np.sum(proportions ** 2)
        inverse_simpson_scores.append(score)

    return {
        "same_source_neighbor_fraction": same_fraction,
        "knn_inverse_simpson_mean": float(np.mean(inverse_simpson_scores)),
    }


def compare_embeddings(ours, reference, n_neighbors=30):
    """Compare our PCs with official harmonypy using common metrics.

    Pearson, RMSE, and kNN overlap are our replication checks,
    not a required three-metric test from the original Harmony paper.
    """
    if ours.shape != reference.shape:
        raise ValueError("Both methods must return the same shape.")

    correlations = []
    for pc_index in range(ours.shape[1]):
        a = ours[:, pc_index]
        b = reference[:, pc_index]
        if np.std(a) == 0 or np.std(b) == 0:
            correlation = np.nan
        else:
            correlation = np.corrcoef(a, b)[0, 1]
        correlations.append(correlation)

    error = ours - reference
    rmse = float(np.sqrt(np.mean(error ** 2)))

    our_neighbors = neighbor_indices(ours, n_neighbors)
    reference_neighbors = neighbor_indices(reference, n_neighbors)
    overlaps = []
    for a, b in zip(our_neighbors, reference_neighbors):
        common = len(set(a) & set(b))
        overlaps.append(common / len(a))

    summary = {
        "mean_pc_pearson": float(np.nanmean(correlations)),
        "rmse": rmse,
        "knn_overlap": float(np.mean(overlaps)),
    }
    per_pc = pd.DataFrame({
        "PC": np.arange(1, len(correlations) + 1),
        "pearson": correlations,
    })
    return summary, per_pc


def run_reference(pcs, metadata, source_key, parameters):
    """Run the official implementation, separate from our own code."""
    import harmonypy
    from importlib.metadata import version

    if version("harmonypy") != "2.1.0":
        raise RuntimeError("Please install harmonypy==2.1.0")

    result = harmonypy.run_harmony(
        pcs, metadata, source_key, **parameters, ncores=1, verbose=False
    )
    if result.Z_corr.shape != pcs.shape:
        raise ValueError("Unexpected reference result shape.")
    return result


def benchmark_methods(pcs, metadata, source_key, parameters, repeats=3):
    """Run each method three times after warmup, then compare runtimes."""
    def run_ours():
        return run_harmony(pcs, metadata, source_key, **parameters, verbose=False)

    def run_official():
        return run_reference(pcs, metadata, source_key, parameters)

    print("  Warming up both methods...", flush=True)
    run_ours()
    run_official()

    times = []
    results = {}
    for repeat in range(repeats):
        # Alternating order helps reduce bias from which method runs first.
        if repeat % 2 == 0:
            order = ["Python Harmony2", "harmonypy 2.1.0"]
        else:
            order = ["harmonypy 2.1.0", "Python Harmony2"]

        for name in order:
            start = perf_counter()
            if name == "Python Harmony2":
                result = run_ours()
            else:
                result = run_official()
            seconds = perf_counter() - start
            results[name] = result
            times.append({"method": name, "repeat": repeat + 1, "seconds": seconds})
            print(f"  {name}: {seconds:.3f} seconds", flush=True)

    return results["Python Harmony2"], results["harmonypy 2.1.0"], pd.DataFrame(times)


def make_umaps(embeddings, random_state=0):
    """Convert each 30-dimensional embedding into two dimensions for plotting."""
    from umap import UMAP

    initial = embeddings["Uncorrected"][:, :2].astype(np.float32)
    initial = initial / max(float(np.std(initial)), 1e-10)

    umaps = {}
    for method, coordinates in embeddings.items():
        print("  UMAP:", method, flush=True)
        model = UMAP(
            n_neighbors=30, min_dist=0.3, metric="euclidean",
            random_state=random_state, n_jobs=1, init=initial.copy(),
        )
        umaps[method] = model.fit_transform(coordinates)
    return umaps


def plot_embedding_comparison(embeddings, umaps, labels, path):
    """Show PCA and UMAP before correction, after ours, and after official."""
    sources = sorted(pd.unique(labels))
    color_for_source = {}
    for i, source in enumerate(sources):
        color_for_source[source] = COLORS[i % len(COLORS)]
    point_colors = [color_for_source[source] for source in labels]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Use a shared axis scale for the PCA panels only.
    all_pcs = np.vstack([coordinates[:, :2] for coordinates in embeddings.values()])
    minimum = all_pcs.min(axis=0)
    maximum = all_pcs.max(axis=0)
    padding = (maximum - minimum) * 0.05

    for column, method in enumerate(METHODS):
        pc_points = embeddings[method][:, :2]
        umap_points = umaps[method]

        axes[0, column].scatter(pc_points[:, 0], pc_points[:, 1], c=point_colors, s=5, alpha=0.65)
        axes[0, column].set_title(method)
        axes[0, column].set_xlabel("PC 1")
        axes[0, column].set_ylabel("PC 2")
        axes[0, column].set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
        axes[0, column].set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])

        axes[1, column].scatter(umap_points[:, 0], umap_points[:, 1], c=point_colors, s=5, alpha=0.65)
        axes[1, column].set_xlabel("UMAP 1")
        axes[1, column].set_ylabel("UMAP 2")
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])

    legend_items = []
    for source in sources:
        item = Line2D([], [], marker="o", linestyle="", color=color_for_source[source], label=source)
        legend_items.append(item)
    fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Liver | {len(labels):,} cells | color = dataset source")
    fig.text(0.5, 0.01, "UMAP fits are separate; absolute positions cannot be directly compared.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.91, bottom=0.15, hspace=0.28, wspace=0.23)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_diagnostics(ours, reference, timing, per_pc, path):
    """Plot convergence, speed, and per-PC agreement with the official method."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")

    axes[0].plot(ours["objective_harmony"], "o-", label="Python Harmony2")
    axes[0].plot(reference.objective_harmony, "o-", label="harmonypy 2.1.0")
    axes[0].set(title="Convergence", xlabel="Iteration", ylabel="Clustering objective")
    axes[0].legend(fontsize=8)

    medians = timing.groupby("method")["seconds"].median()
    axes[1].bar(["Python Harmony2", "harmonypy 2.1.0"],
                [medians["Python Harmony2"], medians["harmonypy 2.1.0"]])
    axes[1].set(title="Runtime", ylabel="Seconds")
    axes[1].tick_params(axis="x", labelrotation=15)

    axes[2].plot(per_pc["PC"], per_pc["pearson"], "o-")
    axes[2].set(title="Agreement with official", xlabel="PC", ylabel="Pearson correlation")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_simulation(random_state=123):
    """Create 900 cells with three known types and three known batch shifts."""
    rng = np.random.default_rng(random_state)
    cell_types = np.repeat(np.arange(3), 300)
    batches = np.tile(np.repeat(np.arange(3), 100), 3)

    centers = np.zeros((3, 12))
    centers[np.arange(3), np.arange(3)] = 7.0
    clean = centers[cell_types] + rng.normal(0, 0.4, (900, 12))

    shifts = np.zeros((3, 12))
    shifts[:, 3:5] = [[-2.5, 1.5], [0.0, -3.0], [2.5, 1.5]]
    observed = clean + shifts[batches]

    metadata = pd.DataFrame({
        "source": ["batch_" + str(batch) for batch in batches],
        "cell_type": cell_types,
    })
    return clean, observed, metadata


def evaluate_simulation(parameters, output_dir):
    """Compare artificial clean data, shifted data, and corrected data."""
    clean, observed, metadata = make_simulation()
    settings = dict(parameters)
    settings["nclust"] = 3

    ours = run_harmony(observed, metadata, "source", **settings, verbose=False)
    reference = run_reference(observed, metadata, "source", settings)

    embeddings = {
        "Clean truth": clean,
        "Uncorrected": observed,
        "Python Harmony2": ours["Z_corr"],
        "harmonypy 2.1.0": reference.Z_corr,
    }

    rows = []
    for method, coordinates in embeddings.items():
        # Remove one global translation before comparing with clean truth.
        centered = coordinates - coordinates.mean(axis=0)
        centered_clean = clean - clean.mean(axis=0)
        rmse = np.sqrt(np.mean((centered - centered_clean) ** 2))
        silhouette = silhouette_score(coordinates, metadata["cell_type"])

        result = {"method": method, "centered_rmse_to_clean": float(rmse),
                  "cell_type_silhouette": float(silhouette)}
        result.update(mixing_metrics(coordinates, metadata["source"]))
        rows.append(result)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "simulation_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex="row", sharey="row", layout="constrained")
    types = metadata["cell_type"].to_numpy()
    batches = pd.factorize(metadata["source"])[0]
    for column, (method, coordinates) in enumerate(embeddings.items()):
        axes[0, column].scatter(coordinates[:, 0], coordinates[:, 1], c=np.asarray(COLORS)[types], s=6, alpha=0.6)
        axes[1, column].scatter(coordinates[:, 3], coordinates[:, 4], c=np.asarray(COLORS)[batches], s=6, alpha=0.6)
        axes[0, column].set(title=method, xlabel="Biology dimension 1", ylabel="Biology dimension 2")
        axes[1, column].set(xlabel="Batch dimension 4", ylabel="Batch dimension 5")
    fig.suptitle("Controlled simulation | top: known cell type | bottom: known batch")
    fig.savefig(output_dir / "simulation.png", dpi=160)
    plt.close(fig)
    return metrics
