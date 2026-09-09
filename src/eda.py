# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def descriptive_statistics(df):
    """
    Calculate descriptive statistics
    for numerical columns.
    """

    numerical_cols = [
        "age",
        "bmi",
        "children",
        "charges"
    ]

    numerical_data = df[numerical_cols]

    statistics = numerical_data.describe()

    median = numerical_data.median()

    iqr = (
        numerical_data.quantile(0.75)
        - numerical_data.quantile(0.25)
    )

    skewness = numerical_data.skew()

    kurtosis = numerical_data.kurtosis()

    return {
        "summary": statistics,
        "median": median,
        "iqr": iqr,
        "skewness": skewness,
        "kurtosis": kurtosis
    }


def create_histograms(df):
    """
    Create distribution plots for numerical variables.
    """

    numerical_cols = ["age", "bmi", "children", "charges"]

    for col in numerical_cols:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=col,
            kde=True
        )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()


def create_scatter_plot(df):
    """
    Create a scatter plot of BMI versus medical charges.
    """

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker"
    )

    plt.title("BMI vs Medical Charges")
    plt.xlabel("BMI")
    plt.ylabel("Medical Charges")

    plt.tight_layout()
    plt.show()


def create_correlation_matrix(df):
    """
    Create a correlation matrix for numerical variables.
    """

    numerical_cols = ["age", "bmi", "children", "charges"]

    correlation = df[numerical_cols].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()
    plt.show()

