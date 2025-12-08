"""
NET Analysis Add-on Module

Provides NET-specific analysis including:
- Age-tier box & whisker plots based on NET tiers.md definitions
- JSON summary with simulation tallies, outliers, and tier compliance validation

Reference: https://github.com/fearandesire/NoEyeTest/blob/main/tiers.md
"""

import math
import json
import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')


# NET Tier Configuration (from tiers.md)
NET_AGE_TIERS = {
    '25-30': {'min_age': 25, 'max_age': 30, 'hard_max': 4},
    '31-34': {'min_age': 31, 'max_age': 34, 'hard_max': 2},
    '35+': {'min_age': 35, 'max_age': 99, 'hard_max': 0},
}

# 80+ OVR Hard Mins by age tier
NET_80_PLUS_HARD_MINS = {
    '25-30': (-2, 0),  # Random between -2 and 0
    '31-34': -10,
    '35+': -14,
}

OVR_CAP = 80


def get_age_tier(age):
    """Get the NET age tier label for a given age."""
    if age < 25:
        return '<25'
    elif 25 <= age <= 30:
        return '25-30'
    elif 31 <= age <= 34:
        return '31-34'
    else:
        return '35+'


def calculate_tier_expected_range(per, age, baseline_ovr):
    """
    Calculate fixed expected min/max progression range for a player.
    
    Validation uses hard bounds from tiers.md and ignores PER. Applies 80+ OVR
    caps per tier (hard max = 0; tier-specific hard mins).
    """
    if age < 25:
        return (0, 0)

    tier = get_age_tier(age)
    try:
        baseline_ovr = float(baseline_ovr)
    except (TypeError, ValueError):
        baseline_ovr = 0
    if math.isnan(baseline_ovr):
        baseline_ovr = 0

    # Fixed hard bounds per tier (PER-independent)
    fixed_bounds = {
        '25-30': (-2, 4),
        '31-34': (-10, 2),
        '35+': (-14, 0),
    }

    # 80+ OVR caps per tiers.md
    hard_cap_bounds = {
        '25-30': (-2, 0),
        '31-34': (-10, 0),
        '35+': (-14, 0),
    }

    if baseline_ovr >= OVR_CAP:
        mn, mx = hard_cap_bounds.get(tier, (0, 0))
        return (int(mn), int(mx))

    mn, mx = fixed_bounds.get(tier, (0, 0))

    # Prevent overshooting OVR cap when approaching 80
    ovr_progression = mx + baseline_ovr
    flag_lower = mn + baseline_ovr
    if ovr_progression >= OVR_CAP:
        mx = max(0, OVR_CAP - baseline_ovr)
        if flag_lower >= OVR_CAP:
            mn = 0

    return (int(mn), int(mx))


def create_net_age_tier_plot(df_path, output_dir):
    """
    Generate NET-specific age tier box & whisker plot.
    
    Creates a boxplot showing Delta distribution grouped by NET age tiers
    (25-30, 31-34, 35+). Saved as plots/net_age_tier_progression.png.
    """
    print("Generating NET age tier plot...")
    
    try:
        df = pd.read_csv(df_path)
    except Exception as e:
        print(f"Error loading CSV for NET plot: {e}")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Add AgeTier column
    df_plot = df.copy()
    df_plot['AgeTier'] = df_plot['Age'].apply(get_age_tier)
    
    # Filter to only include NET-relevant tiers (25+)
    df_plot = df_plot[df_plot['AgeTier'].isin(['25-30', '31-34', '35+'])]
    
    # Set tier order for plot
    tier_order = ['25-30', '31-34', '35+']
    df_plot['AgeTier'] = pd.Categorical(df_plot['AgeTier'], categories=tier_order, ordered=True)
    
    sns.set_theme(style="whitegrid", context="notebook")
    
    plt.figure(figsize=(12, 7))
    
    # Create boxplot with outliers shown
    ax = sns.boxplot(
        x="AgeTier", 
        y="Delta", 
        data=df_plot, 
        hue="AgeTier",
        palette="Set2",
        legend=False,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4}
    )
    
    # Add reference line at 0
    plt.axhline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Add hard max reference lines per tier
    tier_positions = {tier: i for i, tier in enumerate(tier_order)}
    for tier, config in NET_AGE_TIERS.items():
        if tier in tier_positions:
            pos = tier_positions[tier]
            hard_max = config['hard_max']
            plt.hlines(y=hard_max, xmin=pos - 0.4, xmax=pos + 0.4, 
                      colors='green', linestyles=':', linewidth=2, alpha=0.8)
    
    plt.title("NET Age Tier Progression Distribution", fontsize=16)
    plt.xlabel("Age Tier", fontsize=12)
    plt.ylabel("OVR Change (Delta)", fontsize=12)
    
    # Add legend for reference lines
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='Zero Line'),
        Line2D([0], [0], color='green', linestyle=':', linewidth=2, label='Hard Max')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/net_age_tier_progression.png", dpi=300)
    plt.close()
    
    print(f"NET age tier plot saved to {plots_dir}/net_age_tier_progression.png")


def validate_tier_compliance(df):
    """
    Validate each simulation record against expected tier ranges.
    
    Returns dict with violation tallies and details per tier.
    """
    tier_violation_counts = {'25-30': 0, '31-34': 0, '35+': 0}
    
    # Filter to age >= 25
    df_filtered = df[df['Age'] >= 25].copy()
    
    if len(df_filtered) == 0:
        return {
            'summary': {
                'total_violations': 0,
                'by_tier': {tier: {'violations': 0} for tier in tier_violation_counts.keys()}
            },
            'violations': []
        }
    
    # Compute age_tier for the whole column
    df_filtered['age_tier'] = df_filtered['Age'].apply(get_age_tier)
    
    # Drop rows with tiers not in tier_violation_counts
    df_filtered = df_filtered[df_filtered['age_tier'].isin(tier_violation_counts.keys())]
    
    if len(df_filtered) == 0:
        return {
            'summary': {
                'total_violations': 0,
                'by_tier': {tier: {'violations': 0} for tier in tier_violation_counts.keys()}
            },
            'violations': []
        }
    
    # Compute expected_min and expected_max as new columns
    def calculate_expected_range(row):
        per = row.get('PER', 0) if pd.notna(row.get('PER')) else 0
        age = row['Age']
        baseline = row['Baseline']
        return calculate_tier_expected_range(per, age, baseline)
    
    expected_ranges = df_filtered.apply(calculate_expected_range, axis=1)
    df_filtered['expected_min'] = expected_ranges.apply(lambda x: x[0])
    df_filtered['expected_max'] = expected_ranges.apply(lambda x: x[1])
    
    # Build boolean masks for exceeded_max and below_min
    exceeded_max = df_filtered['Delta'] > df_filtered['expected_max']
    below_min = df_filtered['Delta'] < df_filtered['expected_min']
    has_violation = exceeded_max | below_min
    
    # Increment tier_violation_counts by grouping/counting
    violation_counts = df_filtered[has_violation].groupby('age_tier').size()
    for tier, count in violation_counts.items():
        if tier in tier_violation_counts:
            tier_violation_counts[tier] = int(count)
    
    # Construct violations by selecting rows where either mask is True
    # Set violation_type before filtering (exceeded_max takes priority like original elif logic)
    df_filtered.loc[exceeded_max, 'violation_type'] = 'exceeded_max'
    df_filtered.loc[below_min & ~exceeded_max, 'violation_type'] = 'below_min'
    violations_df = df_filtered[has_violation].copy()
    
    # Convert to dict records
    violations = violations_df.apply(lambda row: {
        'name': row.get('Name', 'Unknown'),
        'team': row.get('Team', 'Unknown'),
        'age': int(row['Age']),
        'age_tier': row['age_tier'],
        'baseline': float(row['Baseline']),
        'per': float(row.get('PER', 0)) if pd.notna(row.get('PER')) else 0,
        'actual_delta': float(row['Delta']),
        'expected_min': int(row['expected_min']),
        'expected_max': int(row['expected_max']),
        'violation_type': row['violation_type'],
        'run': int(row.get('Run', 0))
    }, axis=1).tolist()
    
    total_violations = len(violations)
    if total_violations > 100:
        print(f"Note: Truncating violations list from {total_violations} to 100 for readability")
    
    return {
        'summary': {
            'total_violations': total_violations,
            'by_tier': {tier: {'violations': count} for tier, count in tier_violation_counts.items()}
        },
        'violations': violations[:100]
    }


def identify_outliers(df):
    """
    Identify extreme deltas and IQR-based outliers.
    
    Returns dict with extreme cases and outlier lists per tier.
    """
    def extract_player_details(row):
        return {
            'name': row.get('Name', 'Unknown'),
            'team': row.get('Team', 'Unknown'),
            'age': int(row['Age']),
            'baseline': float(row['Baseline']),
            'delta': float(row['Delta']),
            'pct_change': float(row.get('PctChange', 0)) * 100,
            'per': float(row.get('PER', 0)) if pd.notna(row.get('PER')) else 0,
            'run': int(row.get('Run', 0))
        }
    
    # Guard against empty DataFrame or missing 'Delta' column
    if df.empty or 'Delta' not in df.columns:
        return {
            'extremes': {
                'largest_positive_delta': None,
                'largest_negative_delta': None
            },
            'by_age_tier': {'25-30': [], '31-34': [], '35+': []},
            'overall_count': 0
        }
    
    # Overall extremes
    try:
        max_delta_idx = df['Delta'].idxmax()
        min_delta_idx = df['Delta'].idxmin()
        extremes = {
            'largest_positive_delta': extract_player_details(df.loc[max_delta_idx]),
            'largest_negative_delta': extract_player_details(df.loc[min_delta_idx])
        }
    except (ValueError, KeyError):
        extremes = {
            'largest_positive_delta': None,
            'largest_negative_delta': None
        }
    
    # IQR-based outliers per age tier
    df_with_tier = df.copy()
    df_with_tier['AgeTier'] = df_with_tier['Age'].apply(get_age_tier)
    
    outliers_by_tier = {}
    overall_outliers = []
    
    for tier in ['25-30', '31-34', '35+']:
        tier_df = df_with_tier[df_with_tier['AgeTier'] == tier]
        if len(tier_df) == 0:
            outliers_by_tier[tier] = []
            continue
        
        q1 = tier_df['Delta'].quantile(0.25)
        q3 = tier_df['Delta'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        tier_outliers = tier_df[(tier_df['Delta'] < lower_bound) | (tier_df['Delta'] > upper_bound)]
        
        outlier_records = []
        for _, row in tier_outliers.head(25).iterrows():  # Limit per tier
            outlier_records.append(extract_player_details(row))
        
        outliers_by_tier[tier] = outlier_records
        overall_outliers.extend(outlier_records)
    
    return {
        'extremes': extremes,
        'by_age_tier': outliers_by_tier,
        'overall_count': len(overall_outliers)
    }


def generate_net_summary(outputs_csv_path, raw_dir, metadata):
    """
    Generate comprehensive NET analysis summary as JSON.
    
    Creates net_summary.json with:
    - Run info
    - Population stats
    - Delta stats by tier
    - Extremes and outliers
    - God progression summary
    - Tier compliance validation
    """
    print("Generating NET summary...")
    
    try:
        df = pd.read_csv(outputs_csv_path)
    except Exception as e:
        print(f"Error loading CSV for NET summary: {e}")
        return
    
    output_dir = os.path.dirname(outputs_csv_path)
    
    # Add age tier column
    df['AgeTier'] = df['Age'].apply(get_age_tier)
    
    # Run info
    run_info = {
        'runs': int(df['Run'].nunique()) if 'Run' in df.columns else 0,
        'master_seed': metadata.get('master_seed', 'unknown'),
        'timestamp': metadata.get('timestamp', 'unknown'),
        'league_name': metadata.get('league_name', 'unknown'),
        'total_records': len(df)
    }
    
    # Population stats
    unique_players = df.drop_duplicates(subset=['Name']) if 'Name' in df.columns else df
    age_tier_counts = df['AgeTier'].value_counts().to_dict()
    ovr_80_plus_count = len(df[df['Baseline'] >= 80])
    
    population = {
        'total_players': int(unique_players['Name'].nunique()) if 'Name' in df.columns else len(unique_players),
        'total_teams': int(df['Team'].nunique()) if 'Team' in df.columns else 0,
        'age_tiers': {
            '25-30': {'count': int(age_tier_counts.get('25-30', 0))},
            '31-34': {'count': int(age_tier_counts.get('31-34', 0))},
            '35+': {'count': int(age_tier_counts.get('35+', 0))},
            '<25': {'count': int(age_tier_counts.get('<25', 0))}
        },
        'ovr_80_plus': {'count': int(ovr_80_plus_count)}
    }
    
    # Delta stats
    def calc_delta_stats(data):
        if len(data) == 0:
            return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': round(float(data['Delta'].mean()), 3),
            'median': round(float(data['Delta'].median()), 3),
            'std': round(float(data['Delta'].std()), 3),
            'min': float(data['Delta'].min()),
            'max': float(data['Delta'].max())
        }
    
    delta_stats = {
        'overall': calc_delta_stats(df),
        'by_age_tier': {
            '25-30': calc_delta_stats(df[df['AgeTier'] == '25-30']),
            '31-34': calc_delta_stats(df[df['AgeTier'] == '31-34']),
            '35+': calc_delta_stats(df[df['AgeTier'] == '35+'])
        }
    }
    
    # Outliers and extremes
    outlier_data = identify_outliers(df)
    
    # God progression summary
    god_prog_summary = {
        'total_count': 0,
        'max_age': 0,
        'superlucky_players': {}
    }
    
    try:
        godprogs_path = os.path.join(raw_dir, 'godprogs.json')
        superlucky_path = os.path.join(raw_dir, 'superlucky.json')
        
        if os.path.exists(godprogs_path):
            with open(godprogs_path, 'r', encoding='utf-8') as f:
                godprogs = json.load(f)
                god_prog_summary['total_count'] = len(godprogs)
                if godprogs:
                    max_age = max(int(gp.get('Age', 0)) for gp in godprogs)
                    god_prog_summary['max_age'] = max_age
        
        if os.path.exists(superlucky_path):
            with open(superlucky_path, 'r', encoding='utf-8') as f:
                superlucky = json.load(f)
                god_prog_summary['superlucky_players'] = superlucky
    except Exception as e:
        print(f"Warning: Could not load god prog data: {e}")
    
    # Tier compliance validation
    compliance = validate_tier_compliance(df)
    
    # Build final summary
    summary = {
        'run_info': run_info,
        'population': population,
        'delta_stats': delta_stats,
        'extremes': outlier_data['extremes'],
        'outliers': {
            'overall_count': outlier_data['overall_count'],
            'by_age_tier': outlier_data['by_age_tier']
        },
        'god_progression': god_prog_summary,
        'tier_compliance': compliance
    }
    
    # Save JSON summary
    summary_path = os.path.join(output_dir, 'net_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"NET summary saved to {summary_path}")
    
    # Generate human-readable text summary
    txt_path = os.path.join(output_dir, 'net_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("NET ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"League: {run_info['league_name']}\n")
        f.write(f"Runs: {run_info['runs']}\n")
        f.write(f"Total Records: {run_info['total_records']}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("POPULATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Players: {population['total_players']}\n")
        f.write(f"Teams: {population['total_teams']}\n")
        f.write(f"80+ OVR Records: {population['ovr_80_plus']['count']}\n\n")
        
        f.write("Age Tier Distribution:\n")
        for tier in ['25-30', '31-34', '35+']:
            f.write(f"  {tier}: {population['age_tiers'][tier]['count']} records\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("DELTA STATISTICS\n")
        f.write("-" * 40 + "\n")
        overall = delta_stats['overall']
        f.write(f"Overall: mean={overall['mean']}, median={overall['median']}, std={overall['std']}\n")
        f.write(f"         min={overall['min']}, max={overall['max']}\n\n")
        
        f.write("By Age Tier:\n")
        for tier in ['25-30', '31-34', '35+']:
            ts = delta_stats['by_age_tier'][tier]
            f.write(f"  {tier}: mean={ts['mean']}, median={ts['median']}, range=[{ts['min']}, {ts['max']}]\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("EXTREMES\n")
        f.write("-" * 40 + "\n")
        pos = outlier_data['extremes']['largest_positive_delta']
        neg = outlier_data['extremes']['largest_negative_delta']
        f.write(f"Largest Positive: {pos['name']} ({pos['team']})\n")
        f.write(f"  Age: {pos['age']}, Baseline: {pos['baseline']}, Delta: +{pos['delta']}\n\n")
        f.write(f"Largest Negative: {neg['name']} ({neg['team']})\n")
        f.write(f"  Age: {neg['age']}, Baseline: {neg['baseline']}, Delta: {neg['delta']}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("GOD PROGRESSION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total God Progs: {god_prog_summary['total_count']}\n")
        f.write(f"Max Age God Progged: {god_prog_summary['max_age']}\n")
        if god_prog_summary['superlucky_players']:
            f.write("Superlucky Players (multiple god progs):\n")
            for name, count in list(god_prog_summary['superlucky_players'].items())[:10]:
                f.write(f"  {name}: {count} times\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("TIER COMPLIANCE\n")
        f.write("-" * 40 + "\n")
        f.write("Validation: fixed hard bounds per tiers.md (PER-independent)\n")
        f.write(f"Total Violations: {compliance['summary']['total_violations']}\n")
        f.write("By Tier:\n")
        for tier in ['25-30', '31-34', '35+']:
            count = compliance['summary']['by_tier'][tier]['violations']
            f.write(f"  {tier}: {count} violations\n")
        
        if compliance['violations']:
            f.write("\nSample Violations (first 10):\n")
            for v in compliance['violations'][:10]:
                f.write(f"  {v['name']} (Age {v['age']}, {v['age_tier']}): ")
                f.write(f"Delta={v['actual_delta']}, Expected=[{v['expected_min']}, {v['expected_max']}] ")
                f.write(f"({v['violation_type']})\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"NET text summary saved to {txt_path}")
    
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path)
        raw_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(out_dir, 'raw')
        
        create_net_age_tier_plot(csv_path, out_dir)
        generate_net_summary(csv_path, raw_dir, {})
    else:
        print("Usage: python net_analysis.py <outputs.csv> [output_dir] [raw_dir]")
