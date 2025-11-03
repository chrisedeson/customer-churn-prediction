"""
Customer Lifetime Value (CLV) analysis module.
Computes CLV, creates quartiles, and analyzes churn by segment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, TARGET_COL,
    EXPECTED_TENURE_NON_CHURNER, EXPECTED_TENURE_CHURNER
)


def compute_expected_tenure(churn_prob):
    """
    Compute expected tenure based on churn probability.
    Assumption: Expected tenure = (1 - churn_prob) * 24 + churn_prob * 6
    This gives a weighted average between non-churner (24 months) and churner (6 months) tenure.
    """
    return (1 - churn_prob) * EXPECTED_TENURE_NON_CHURNER + churn_prob * EXPECTED_TENURE_CHURNER


def compute_clv(df, churn_prob_col=None):
    """
    Compute CLV for each customer.
    CLV = MonthlyCharges × ExpectedTenure
    
    Args:
        df: DataFrame with customer data
        churn_prob_col: Column name containing churn probability (0-1).
                       If None, uses actual churn label (0 or 1).
    """
    df = df.copy()
    
    if churn_prob_col and churn_prob_col in df.columns:
        # Use predicted probability for more realistic CLV
        df['expected_tenure'] = compute_expected_tenure(df[churn_prob_col])
    else:
        # Fallback: use actual churn label (0 or 1)
        df['expected_tenure'] = compute_expected_tenure(df[TARGET_COL])
    
    df['clv'] = df['MonthlyCharges'] * df['expected_tenure']
    return df


def create_clv_quartiles(df):
    """Create CLV quartiles (Low/Med/High/Premium)."""
    df = df.copy()
    df['clv_quartile'] = pd.qcut(
        df['clv'],
        q=4,
        labels=['Low', 'Med', 'High', 'Premium']
    )
    return df


def compute_churn_rate_by_quartile(df):
    """Compute churn rate for each CLV quartile."""
    churn_by_quartile = df.groupby('clv_quartile', observed=True)[TARGET_COL].agg([
        ('total', 'count'),
        ('churned', 'sum'),
        ('churn_rate', 'mean')
    ])
    return churn_by_quartile


def analyze_clv_distribution(df):
    """Analyze CLV distribution statistics."""
    clv_stats = df['clv'].describe()
    return clv_stats


def get_clv_insights(churn_by_quartile):
    """
    Generate business insights from CLV analysis.
    Returns list of insight strings.
    """
    insights = []
    
    # Insight 1: Churn rate comparison
    high_premium_churn = churn_by_quartile.loc[['High', 'Premium'], 'churn_rate'].mean()
    low_med_churn = churn_by_quartile.loc[['Low', 'Med'], 'churn_rate'].mean()
    
    insights.append(
        f"High-value customers (High/Premium) have {high_premium_churn:.1%} churn rate "
        f"vs {low_med_churn:.1%} for low-value customers."
    )
    
    # Insight 2: Premium segment specifics
    premium_churn = churn_by_quartile.loc['Premium', 'churn_rate']
    premium_total = churn_by_quartile.loc['Premium', 'total']
    
    insights.append(
        f"Premium segment ({premium_total} customers) has {premium_churn:.1%} churn rate, "
        f"representing the highest revenue at risk."
    )
    
    # Insight 3: Retention priority
    quartile_risk = churn_by_quartile['churned'] * churn_by_quartile.index.map({
        'Low': 1, 'Med': 2, 'High': 3, 'Premium': 4
    })
    priority_quartile = quartile_risk.idxmax()
    
    insights.append(
        f"Prioritize retention efforts on {priority_quartile} CLV segment "
        f"to maximize revenue protection."
    )
    
    return insights


def create_clv_visualizations(df, output_dir=None):
    """Create CLV analysis visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: CLV distribution
    axes[0].hist(df['clv'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Customer Lifetime Value ($)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('CLV Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Churn rate by CLV quartile
    churn_by_quartile = compute_churn_rate_by_quartile(df)
    quartiles = churn_by_quartile.index
    churn_rates = churn_by_quartile['churn_rate'] * 100
    
    bars = axes[1].bar(quartiles, churn_rates, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('CLV Quartile')
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].set_title('Churn Rate by CLV Quartile')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'clv_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    return fig


def run_clv_analysis():
    """Main pipeline for CLV analysis."""
    print("=" * 50)
    print("Starting CLV analysis")
    print("=" * 50)
    
    # Load training data
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(train_df)} training records")
    
    # Compute CLV
    train_df = compute_clv(train_df)
    print(f"\nCLV Statistics:")
    print(analyze_clv_distribution(train_df))
    
    # Create quartiles
    train_df = create_clv_quartiles(train_df)
    
    # Analyze churn by quartile
    churn_by_quartile = compute_churn_rate_by_quartile(train_df)
    print(f"\nChurn Rate by CLV Quartile:")
    print(churn_by_quartile)
    
    # Generate insights
    insights = get_clv_insights(churn_by_quartile)
    print(f"\nKey Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("=" * 50)
    print("CLV analysis complete")
    print("=" * 50)
    
    return train_df, churn_by_quartile, insights


if __name__ == "__main__":
    run_clv_analysis()
