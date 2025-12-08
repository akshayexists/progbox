import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

def create_plots(df_path, output_dir):
    """
    Generate visualizations for the progression simulation results.

    Creates four plots analyzing progression patterns: distribution of OVR change
    by age, average progression trends, talent vs progression scatter, and
    progression by talent tier. All plots are saved as PNG files to the plots
    subdirectory within the specified output directory.

    Parameters
    ----------
    df_path : str
        Path to CSV file or dataframe source containing simulation results.
        Expected columns: 'Age', 'Delta', 'Baseline'.
    output_dir : str
        Directory path where plot files will be written. A 'plots' subdirectory
        will be created if it doesn't exist.

    Returns
    -------
    None
        Function returns None. On successful completion, plot files are written
        to disk. Returns early (None) if CSV loading fails.

    Raises
    ------
    FileNotFoundError
        If the CSV file specified by df_path does not exist.
    ValueError
        If the CSV lacks required columns or contains invalid data.
    IOError
        If the output directory cannot be created or plot files cannot be written.
    """
    print("Generating plots...")
    
    # Load data
    try:
        df = pd.read_csv(df_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Ensure plots dir exists
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set theme - simple and pretty
    sns.set_theme(style="whitegrid", context="notebook")
    
    # --- Plot 1: Distribution of OVR Change by Age ---
    # This addresses "easy viewing of the same age ranges"
    plt.figure(figsize=(14, 7))
    # Boxplot shows median, quartiles, and outliers
    sns.boxplot(x="Age", y="Delta", hue="Age", data=df, palette="viridis", legend=False, flierprops={"marker": "o", "markersize": 2, "alpha": 0.5})
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title("Distribution of OVR Change by Age", fontsize=16)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("OVR Change (Delta)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/dist_ovr_change_by_age.png", dpi=300)
    plt.close()
    
    # --- Plot 2: Average Progression with Confidence Intervals ---
    plt.figure(figsize=(14, 7))
    # Lineplot aggregates by Age automatically
    sns.lineplot(x="Age", y="Delta", data=df, marker="o", errorbar=('ci', 95))
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title("Average Progression Trend by Age (95% CI)", fontsize=16)
    plt.ylabel("Average OVR Change", fontsize=12)
    plt.savefig(f"{plots_dir}/avg_progression_trend.png", dpi=300)
    plt.close()

    # --- Plot 3: Talent vs Progression (Scatter) ---
    # Addresses "talented, vs avg talent"
    # We'll sample if the dataset is huge to avoid overcrowding
    plot_df = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 10000 else df
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="Baseline", y="Delta", hue="Age", data=plot_df, palette="viridis", alpha=0.6, s=30)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Progression vs. Starting OVR (Talent)", fontsize=16)
    plt.xlabel("Starting OVR (Baseline)", fontsize=12)
    plt.ylabel("OVR Change", fontsize=12)
    plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/scatter_talent_vs_progression.png", dpi=300)
    plt.close()
    
    # --- Plot 4: Progression by Talent Tier ---
    # Binning baseline OVR into tiers
    df_plot = df.copy()
    bins = [0, 40, 50, 60, 70, 100]
    labels = ['<40 (Low)', '40-50 (Avg)', '50-60 (Good)', '60-70 (Star)', '70+ (God)']
    df_plot['TalentTier'] = pd.cut(df_plot['Baseline'], bins=bins, labels=labels)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="TalentTier", y="Delta", hue="Age", data=df_plot, palette="coolwarm", showfliers=False)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title("Progression Distribution by Talent Tier and Age", fontsize=16)
    plt.xlabel("Talent Tier (Starting OVR)", fontsize=12)
    plt.ylabel("OVR Change", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., title="Age")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/box_talent_tier_progression.png", dpi=300)
    plt.close()

    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Assume first arg is csv path, second (optional) is output dir
        csv_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path)
        create_plots(csv_path, out_dir)
    else:
        print("Usage: python visualize.py <path_to_outputs.csv> [output_dir]")
