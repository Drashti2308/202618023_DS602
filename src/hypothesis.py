# src/hypothesis.py

import pandas as pd
from scipy import stats


def compare_two_groups(df, group_col, value_col):
    """
    Compare two groups using:
    Shapiro-Wilk
    Levene's test
    t-test / Mann-Whitney U
    """

    # Get the unique groups
    groups = df[group_col].dropna().unique()

    # Check that exactly two groups are available
    if len(groups) != 2:
        raise ValueError("This test requires exactly two groups.")

    # Split the numerical values into two groups
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()

    # Shapiro-Wilk normality test
    shapiro1 = stats.shapiro(group1)
    shapiro2 = stats.shapiro(group2)

    # Levene's equal variance test
    levene_result = stats.levene(group1, group2)

    # Check normality
    normal1 = shapiro1.pvalue > 0.05
    normal2 = shapiro2.pvalue > 0.05

    # Choose the appropriate test
    if normal1 and normal2:
        test_name = "Two-Sample t-Test"

        test_result = stats.ttest_ind(
            group1,
            group2,
            equal_var=levene_result.pvalue > 0.05
        )

    else:
        test_name = "Mann-Whitney U Test"

        test_result = stats.mannwhitneyu(
            group1,
            group2,
            alternative="two-sided"
        )

    # Final conclusion
    if test_result.pvalue < 0.05:
        conclusion = "Reject H0"
    else:
        conclusion = "Fail to Reject H0"

    return {
        "groups": groups,
        "shapiro_group1": shapiro1,
        "shapiro_group2": shapiro2,
        "levene": levene_result,
        "test_name": test_name,
        "test_result": test_result,
        "conclusion": conclusion
    }

def run_anova(df, group_col, value_col):
    """
    Perform One-Way ANOVA.
    """

    # Get the unique groups
    groups = df[group_col].dropna().unique()

    # Create a list containing the values
    # for each group
    group_data = [
        df[df[group_col] == group][value_col].dropna()
        for group in groups
    ]

    # Perform One-Way ANOVA
    anova_result = stats.f_oneway(*group_data)

    # Extract the p-value
    p_value = anova_result.pvalue

    # Make the final decision
    if p_value < 0.05:
        conclusion = "Reject H0"
    else:
        conclusion = "Fail to Reject H0"

    return {
        "groups": groups,
        "anova_result": anova_result,
        "conclusion": conclusion
    }

