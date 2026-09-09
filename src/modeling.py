# src/modeling.py

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
def prepare_data(df):
    """
    Prepare numerical and categorical variables
    for regression.
    """

    # Select predictors and target
    X = df[["age", "bmi", "children", "smoker", "region"]]
    y = df["charges"]

    # Convert categorical variables into dummy variables
    X = pd.get_dummies(
        X,
        columns=["smoker", "region"],
        drop_first=True
    )

    # Convert boolean columns to integers
    X = X.astype(float)

    # Add intercept
    X = sm.add_constant(X)

    return X, y


def build_model(df):
    """
    Build the multiple linear regression model
    using OLS.
    """

    X, y = prepare_data(df)

    # Fit OLS regression
    model = sm.OLS(y, X).fit()

    return model, X, y


def calculate_vif(X):
    """
    Calculate VIF for regression predictors.
    """

    # Remove the intercept
    X_without_const = X.drop(columns=["const"])

    # Create a DataFrame for VIF results
    vif_data = pd.DataFrame()

    vif_data["Variable"] = X_without_const.columns

    vif_data["VIF"] = [
        variance_inflation_factor(
            X_without_const.values,
            i
        )
        for i in range(X_without_const.shape[1])
    ]

    return vif_data

def create_diagnostic_plots(model):
    """
    Create regression diagnostic plots and perform
    the Jarque-Bera normality test.
    """

    fitted_values = model.fittedvalues
    residuals = model.resid

    # Create two plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------------------------------------------------
    # Plot 1: Residuals vs Fitted Values
    # -------------------------------------------------
    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        ax=axes[0]
    )

    axes[0].axhline(
        y=0,
        linestyle="--"
    )

    axes[0].set_title("Residuals vs Fitted Values")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")

    # -------------------------------------------------
    # Plot 2: Q-Q Plot
    # -------------------------------------------------
    sm.qqplot(
        residuals,
        line="45",
        ax=axes[1]
    )

    axes[1].set_title("Q-Q Plot of Residuals")

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # Jarque-Bera Test
    # -------------------------------------------------
    jb_result = stats.jarque_bera(residuals)

    return jb_result