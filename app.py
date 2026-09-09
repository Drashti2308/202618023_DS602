import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.eda import (
    descriptive_statistics,
    create_histograms,
    create_scatter_plot,
    create_correlation_matrix
)

from src.hypothesis import (
    compare_two_groups,
    run_anova
)

from src.modeling import (
    build_model,
    calculate_vif
)


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Insurance Data Science Dashboard",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance.csv")
    return df


df = load_data()


# -------------------------------------------------
# Title and Introduction
# -------------------------------------------------

st.title("📊 Medical Insurance Data Science Dashboard")

st.write(
    """
    This dashboard performs exploratory data analysis,
    statistical hypothesis testing, and multiple linear
    regression on the medical insurance dataset.
    """
)

st.write("### Dataset Preview")
st.dataframe(df.head())


# -------------------------------------------------
# Create Tabs
# -------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Data Exploration",
        "🧪 Hypothesis Testing Lab",
        "🔮 Live Prediction & Diagnostics"
    ]
)


# =================================================
# TAB 1: DATA EXPLORATION
# =================================================

with tab1:

    st.header("Data Exploration")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of Records", df.shape[0])

    with col2:
        st.metric("Number of Variables", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Descriptive Statistics")

    descriptive_result = descriptive_statistics(df)

    st.write("Summary Statistics")
    st.dataframe(descriptive_result["summary"])

    st.write("Median")
    st.dataframe(descriptive_result["median"].to_frame("Median"))

    st.write("Interquartile Range")
    st.dataframe(descriptive_result["iqr"].to_frame("IQR"))

    st.write("Skewness")
    st.dataframe(descriptive_result["skewness"].to_frame("Skewness"))

    st.write("Kurtosis")
    st.dataframe(descriptive_result["kurtosis"].to_frame("Kurtosis"))

    st.subheader("Histograms with KDE")

    numerical_columns = [
        "age",
        "bmi",
        "children",
        "charges"
    ]

    selected_column = st.selectbox(
        "Select a numerical variable",
        numerical_columns
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=selected_column,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {selected_column}")
    st.pyplot(fig)

    st.subheader("Scatter Plot")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker",
        ax=ax
    )

    ax.set_title("BMI vs Medical Charges")
    st.pyplot(fig)

    st.subheader("Correlation Matrix")

    numerical_df = df.select_dtypes(include="number")
    correlation_matrix = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Matrix")
    st.pyplot(fig)


# =================================================
# TAB 2: HYPOTHESIS TESTING LAB
# =================================================

with tab2:

    st.header("Hypothesis Testing Lab")

    st.subheader("Two-Group Comparison")

    st.write(
        """
        This section compares medical charges between two groups.
        The Shapiro-Wilk test checks normality, and Levene's test
        checks equality of variances. Depending on the assumptions,
        either an independent-samples t-test or Mann-Whitney U test
        is used.
        """
    )

    group_column = st.selectbox(
        "Select grouping variable",
        ["smoker", "sex"]
    )

    value_column = st.selectbox(
        "Select numerical variable",
        ["charges", "bmi", "age"]
    )

    if st.button("Run Two-Group Test"):

        result = compare_two_groups(
            df,
            group_column,
            value_column
        )

        st.subheader("Test Result")

        st.write(result)

    st.subheader("One-Way ANOVA")

    st.write(
        """
        One-Way ANOVA evaluates whether the mean medical charges
        differ significantly across the four geographical regions.
        """
    )

    if st.button("Run ANOVA"):

        anova_result = run_anova(
            df,
            "region",
            "charges"
        )

        st.write(anova_result)


# =================================================
# TAB 3: LIVE PREDICTION AND DIAGNOSTICS
# =================================================

with tab3:

    st.header("Live Prediction & Diagnostics")

    st.subheader("Multiple Linear Regression Model")

    model, X, y = build_model(df)

    st.write(
        """
        The regression model predicts medical insurance charges
        using age, BMI, number of children, smoking status,
        and geographical region.
        """
    )

    

    st.subheader("Regression Summary")

    model, X, y = build_model(df)

# ---------------------------------------------
# Model Information
# ---------------------------------------------

    st.write("Model Information")

    summary_tables = model.summary().tables

    model_information = pd.DataFrame(
        summary_tables[0].data
    )

    st.dataframe(
        model_information,
        use_container_width=True
    )

    # ---------------------------------------------
    # Estimated Coefficients
    # ---------------------------------------------

    st.write("Estimated Coefficients")

    coefficient_table = pd.DataFrame(
        summary_tables[1].data
    )

    coefficient_table.columns = coefficient_table.iloc[0]

    coefficient_table = coefficient_table.iloc[1:]

    coefficient_table = coefficient_table.reset_index(drop=True)

    st.dataframe(
        coefficient_table,
        use_container_width=True
    )

    # ---------------------------------------------
    # Coefficient Details
    # ---------------------------------------------

    st.write("Regression Coefficient Details")

    coefficient_details = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values,
            "Lower 95% CI": model.conf_int()[0].values,
            "Upper 95% CI": model.conf_int()[1].values
        }
    )

    st.dataframe(
        coefficient_details,
        use_container_width=True
    )

    # ---------------------------------------------
    # Model Performance
    # ---------------------------------------------

    st.write("Model Performance")

    performance_table = pd.DataFrame(
        {
            "Metric": [
                "R-squared",
                "Adjusted R-squared",
                "F-statistic",
                "F-statistic p-value",
                "Number of Observations"
            ],
            "Value": [
                model.rsquared,
                model.rsquared_adj,
                model.fvalue,
                model.f_pvalue,
                int(model.nobs)
            ]
        }
    )

    st.dataframe(
        performance_table,
        use_container_width=True
    )

    st.subheader("Variance Inflation Factor")

    vif_result = calculate_vif(X)

    st.dataframe(vif_result)

    st.subheader("Enter Patient Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=70.0,
        value=25.0
    )

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

    if st.button("Predict Medical Charges"):

        input_data = pd.DataFrame(
            {
                "age": [age],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region]
            }
        )

        input_data = pd.get_dummies(
            input_data,
            columns=["smoker", "region"],
            drop_first=True
        )

        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )

        input_data = input_data.astype(float)

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Medical Charges: ${prediction.iloc[0]:,.2f}"
        )