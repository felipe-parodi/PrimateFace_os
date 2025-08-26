"""Visualize evaluation results from model predictions.

This script creates comprehensive visualizations of model evaluation results
including confusion matrices, PR curves, and error distributions.

Example:
    $ python visualize_eval_results.py --results eval_results.json --output viz/
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# --- Configuration ---
RESULTS_JSON_PATH = Path(__file__).resolve().parent / "evaluation_results.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_visualizations"
LABEL_CSV = Path(__file__).resolve().parent.parent / "path/to/genera.csv"

# Plotting parameters
PLT_FIGSIZE = (12, 7)
DEFAULT_COLOR = "grey"
SUPERFAMILY_ORDER = [
    "lemuroidea",
    "lorisoidea",
    "tarsioidea",
    "ceboidea",
    "cercopithecoidea",
    "hominoidea",
]
plt.rcParams["svg.fonttype"] = "none"

def load_and_prepare_data(results_path: Path) -> Optional[pl.DataFrame]:
    """Loads evaluation results and flattens them into a Polars DataFrame.
    
    Args:
        results_path: Path to JSON file with evaluation results
        
    Returns:
        DataFrame with flattened results, or None if loading fails
    """
    print(f"Loading results from {results_path}...")
    try:
        with open(results_path, "r") as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return None

    flat_results = []
    for sf, genera in results_data.items():
        for genus, metrics in genera.items():
            if metrics:  # Ensure there are metrics to report
                flat_results.append(
                    {"superfamily": sf, "genus": genus, **metrics}
                )

    if not flat_results:
        print("No results found in the JSON file.")
        return None

    df = pl.DataFrame(flat_results)
    # The superfamily name in the CSV might be Platyrrhini, but user requested Ceboidea
    df = df.with_columns(
        pl.when(pl.col("superfamily") == "platyrrhini")
        .then(pl.lit("ceboidea"))
        .otherwise(pl.col("superfamily"))
        .alias("superfamily")
    )
    print("Data prepared successfully.")
    return df


def generate_color_map(superfamily_list: List[str]) -> Dict[str, np.ndarray]:
    """Generates a color map for the given list of superfamilies.
    
    Args:
        superfamily_list: List of superfamily names
        
    Returns:
        Dictionary mapping superfamily names to colors
    """
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(superfamily_list)))
    color_map = {sf.lower(): color for sf, color in zip(superfamily_list, colors)}
    color_map["unknown"] = DEFAULT_COLOR
    return color_map


def plot_superfamily_metrics(df: pl.DataFrame, output_dir: Path, color_map: Dict[str, np.ndarray]) -> None:
    """Plots mAP and NME averaged by superfamily.
    
    Args:
        df: DataFrame with evaluation results
        output_dir: Directory to save plots
        color_map: Mapping from superfamily to color
    """
    print("Generating superfamily summary plots...")
    
    # Filter out 'unknown' before aggregating
    df_filtered = df.filter(pl.col("superfamily") != "unknown")

    sf_metrics = (
        df_filtered.group_by("superfamily")
        .agg([pl.mean("mAP").alias("avg_mAP"), pl.mean("NME").alias("avg_NME")])
    )

    # Apply custom sort order
    sf_metrics = sf_metrics.with_columns(
        pl.col("superfamily").cast(pl.Enum(SUPERFAMILY_ORDER))
    ).sort("superfamily")

    # Plot Average mAP
    fig_map, ax_map = plt.subplots(figsize=PLT_FIGSIZE)
    colors_map = [color_map.get(sf, DEFAULT_COLOR) for sf in sf_metrics["superfamily"]]
    ax_map.bar(sf_metrics["superfamily"], sf_metrics["avg_mAP"], color=colors_map)
    ax_map.set_title("Average mAP (IoU=0.50:0.95) per Superfamily (↑)")
    ax_map.set_ylabel("mAP@.50:.95 (bbox)")
    ax_map.set_xlabel("Superfamily")
    
    # Focus y-axis
    min_val = sf_metrics["avg_mAP"].min()
    max_val = sf_metrics["avg_mAP"].max()
    y_min = max(0, min_val - 0.1) if min_val is not None else 0
    y_max = min(1, max_val + 0.1) if max_val is not None else 1
    ax_map.set_ylim(y_min, y_max)

    plt.xticks(rotation=45, ha="right")
    fig_map.tight_layout()
    map_path = output_dir / "superfamily_avg_mAP.svg"
    fig_map.savefig(map_path)
    print(f"Saved superfamily mAP plot to {map_path}")

    # Plot Average NME
    fig_nme, ax_nme = plt.subplots(figsize=PLT_FIGSIZE)
    colors_nme = [color_map.get(sf, DEFAULT_COLOR) for sf in sf_metrics["superfamily"]]
    ax_nme.bar(sf_metrics["superfamily"], sf_metrics["avg_NME"], color=colors_nme)
    ax_nme.set_title("Average NME per Superfamily (↓)")
    ax_nme.set_ylabel("NME (lower is better)")
    ax_nme.set_xlabel("Superfamily")
    plt.xticks(rotation=45, ha="right")
    fig_nme.tight_layout()
    nme_path = output_dir / "superfamily_avg_NME.svg"
    fig_nme.savefig(nme_path)
    print(f"Saved superfamily NME plot to {nme_path}")

    plt.close("all")


def plot_genus_metrics(df: pl.DataFrame, output_dir: Path, color_map: Dict[str, np.ndarray], order_by: str = "alphabetical") -> None:
    """Plots mAP and NME for each genus, with specified ordering.
    
    Args:
        df: DataFrame with evaluation results
        output_dir: Directory to save plots
        color_map: Mapping from superfamily to color
        order_by: Ordering method ('alphabetical' or 'superfamily')
    """
    
    # Filter out 'unknown' at the beginning to apply to all ordering types
    df = df.filter(pl.col("superfamily") != "unknown")

    if order_by == "alphabetical":
        print("Generating genus plots (alphabetical order)...")
        df_sorted = df.sort("genus")
        file_suffix = "alpha"
    elif order_by == "superfamily":
        print("Generating genus plots (superfamily order)...")
        df_sorted = df.with_columns(
            pl.col("superfamily").cast(
                pl.Enum(SUPERFAMILY_ORDER)
            )
        ).sort(["superfamily", "genus"])
        file_suffix = "superfamily_order"
    else:
        raise ValueError("`order_by` must be 'alphabetical' or 'superfamily'")

    # --- Plot Genus mAP ---
    df_map = df_sorted.filter(pl.col("mAP").is_not_null())
    if not df_map.is_empty():
        fig_map, ax_map = plt.subplots(figsize=PLT_FIGSIZE)
        colors = [color_map.get(sf, DEFAULT_COLOR) for sf in df_map["superfamily"]]
        ax_map.bar(df_map["genus"], df_map["mAP"], color=colors)
        ax_map.set_title(f"mAP (IoU=0.50:0.95) by Genus (Ordered by {order_by.capitalize()}) (↑)")
        ax_map.set_ylabel("mAP@.50:.95 (bbox)")

        # Focus y-axis
        min_map, max_map = df_map["mAP"].min(), df_map["mAP"].max()
        y_min = max(0, min_map - 0.1) if min_map is not None else 0
        y_max = min(1, max_map + 0.1) if max_map is not None else 1
        ax_map.set_ylim(y_min, y_max)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for sf, c in sorted(color_map.items())
            if sf in df["superfamily"].unique().to_list()
        ]
        labels = [
            sf.capitalize()
            for sf, c in sorted(color_map.items())
            if sf in df["superfamily"].unique().to_list()
        ]
        ax_map.legend(handles, labels, title="Superfamily")

        plt.xticks(rotation=45, ha="right")
        fig_map.tight_layout()
        map_path = output_dir / f"genus_mAP_{file_suffix}.svg"
        fig_map.savefig(map_path)
        print(f"Saved genus mAP plot to {map_path}")

    # --- Plot Genus NME ---
    df_nme = df_sorted.filter(pl.col("NME").is_not_nan())
    if not df_nme.is_empty():
        fig_nme, ax_nme = plt.subplots(figsize=PLT_FIGSIZE)
        colors = [color_map.get(sf, DEFAULT_COLOR) for sf in df_nme["superfamily"]]
        ax_nme.bar(df_nme["genus"], df_nme["NME"], color=colors)
        ax_nme.set_title(f"NME by Genus (Ordered by {order_by.capitalize()}) (↓)")
        ax_nme.set_ylabel("NME (lower is better)")

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for sf, c in sorted(color_map.items())
            if sf in df["superfamily"].unique().to_list()
        ]
        labels = [
            sf.capitalize()
            for sf, c in sorted(color_map.items())
            if sf in df["superfamily"].unique().to_list()
        ]
        ax_nme.legend(handles, labels, title="Superfamily")

        plt.xticks(rotation=45, ha="right")
        fig_nme.tight_layout()
        nme_path = output_dir / f"genus_NME_{file_suffix}.svg"
        fig_nme.savefig(nme_path)
        print(f"Saved genus NME plot to {nme_path}")

    plt.close("all")


def main() -> None:
    """Main execution pipeline for generating visualizations."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["svg.fonttype"] = "none"

    df = load_and_prepare_data(RESULTS_JSON_PATH)
    if df is None:
        return

    # Generate color map from original CSV to maintain consistency
    try:
        superfamily_df = pl.read_csv(LABEL_CSV)
        superfamilies = superfamily_df["Superfamily"].unique().drop_nulls().to_list()
        color_map = generate_color_map(superfamilies)
    except FileNotFoundError:
        print(f"Warning: Superfamily CSV not found at {LABEL_CSV}. Using unique values from results.")
        superfamilies = df["superfamily"].unique().to_list()
        color_map = generate_color_map(superfamilies)
    
    # Generate and save all plots
    plot_superfamily_metrics(df, OUTPUT_DIR, color_map)
    plot_genus_metrics(df, OUTPUT_DIR, color_map, order_by="alphabetical")
    plot_genus_metrics(df, OUTPUT_DIR, color_map, order_by="superfamily")

    print("\nVisualization script finished.")


if __name__ == "__main__":
    main() 