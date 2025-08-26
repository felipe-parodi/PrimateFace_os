"""Plot distribution of samples across primate genera.

This script creates visualizations showing the distribution of training/test
samples across different primate genera for dataset analysis.

Example:
    $ python plot_genus_distribution.py --annotations train.json --output genus_dist.png
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

# Configuration
COUNT_CSV = Path(__file__).resolve().parent / "genus_count.csv"
LABEL_CSV = Path(__file__).resolve().parent.parent / "path/to/genera.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_visualizations"
OUTPUT_SVG = OUTPUT_DIR / "genus_distribution_with_superfamily_pie.svg"

# Plotting parameters from visualize_eval_results.py for consistency
PLT_FIGSIZE = (12, 8)
DEFAULT_COLOR = "grey"
SUPERFAMILY_ORDER = [
    "lemuroidea",
    "lorisoidea",
    "tarsioidea",
    "ceboidea",
    "cercopithecoidea",
    "hominoidea",
]

def load_and_prepare_data(count_path: Path, label_path: Path) -> Optional[pl.DataFrame]:
    """Loads and merges genus counts with superfamily information.
    
    Args:
        count_path: Path to genus count CSV file
        label_path: Path to superfamily labels CSV file
        
    Returns:
        Merged and prepared DataFrame, or None if loading fails
    """
    print("Loading and preparing data...")
    try:
        count_df = pl.read_csv(count_path).filter(pl.col("genus").is_not_null())
        label_df = pl.read_csv(label_path)
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return None

    # Clean and merge data
    label_df = label_df.select(["Genus", "Superfamily"]).filter(pl.col("Genus").is_not_null()).unique(subset=["Genus"])
    
    # Use case-insensitive inner join to keep only matching genera
    data_df = count_df.with_columns(
        pl.col("genus").str.to_lowercase().alias("genus_lower")
    ).join(
        label_df.with_columns(pl.col("Genus").str.to_lowercase().alias("genus_lower")),
        on="genus_lower",
        how="inner"
    ).drop("genus_lower", "Genus").with_columns(
        pl.col("Superfamily").str.to_lowercase().alias("superfamily")
    )

    data_df = data_df.with_columns(
        pl.col("superfamily").cast(pl.Enum(SUPERFAMILY_ORDER))
    ).sort(["superfamily", "genus"])
    
    print("Data prepared successfully.")
    return data_df


def plot_distribution(df: Optional[pl.DataFrame], color_map: Dict[str, Any], output_path: Path) -> None:
    """Plots the histogram of image counts by genus and an inset pie chart by superfamily.
    
    Args:
        df: DataFrame with genus counts and superfamily information
        color_map: Mapping from superfamily to color
        output_path: Path to save the plot
        
    Returns:
        None
    """
    if df is None or df.is_empty():
        print("No data to plot.")
        return

    print("Generating plots...")
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=PLT_FIGSIZE)

    # Main Bar Chart: Genus Distribution
    colors = [color_map.get(sf, DEFAULT_COLOR) for sf in df["superfamily"]]
    ax.bar(df["genus"], df["count"], color=colors)
    
    ax.set_title("Image Count per Genus (Color-coded by Superfamily)", fontweight='bold', fontsize=16)
    ax.set_ylabel("Image Count")
    ax.set_xlabel("Genus")
    plt.xticks(rotation=45, ha="right")

    min_count = df["count"].min()
    max_count = df["count"].max()
    ax.set_ylim(max(0, min_count - 50), max_count + 50)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for sf, c in color_map.items() if sf != 'unknown' and sf in df["superfamily"].unique().to_list()]
    labels = [sf.capitalize() for sf, c in color_map.items() if sf != 'unknown' and sf in df["superfamily"].unique().to_list()]
    ax.legend(handles, labels, title="Superfamily", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Inset Pie Chart: Superfamily Distribution
    superfamily_counts = df.group_by("superfamily").agg(pl.sum("count")).sort("superfamily")
    pie_colors = [color_map.get(sf, DEFAULT_COLOR) for sf in superfamily_counts["superfamily"]]
    
    inset_ax = fig.add_axes([0.65, 0.65, 0.2, 0.2]) # [left, bottom, width, height]
    wedges, texts, autotexts = inset_ax.pie(
        superfamily_counts["count"], 
        autopct='%1.1f%%', 
        colors=pie_colors,
        pctdistance=0.85,
        textprops={'color':"w", 'fontsize': 8, 'weight': 'bold'}
    )
    
    inset_ax.set_title("Superfamily Distribution", fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save Figure
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main() -> None:
    """Main execution pipeline."""
    df = load_and_prepare_data(COUNT_CSV, LABEL_CSV)
    if df is not None:
        # Hard-code the color map to match visualize_eval_results.py output using indices from the tab20 color reference chart.
        cmap = plt.get_cmap("tab20")
        color_map = {
            "lemuroidea": cmap.colors[0],
            "lorisoidea": cmap.colors[16],
            "tarsioidea": cmap.colors[4],
            "ceboidea": cmap.colors[12],
            "cercopithecoidea": cmap.colors[8],
            "hominoidea": cmap.colors[19],
            "unknown": DEFAULT_COLOR
        }
        plot_distribution(df, color_map, OUTPUT_SVG)
    
    print("\nScript finished.")


if __name__ == "__main__":
    main() 