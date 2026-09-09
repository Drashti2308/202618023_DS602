deployed app link: https://boardinsurance.streamlit.app/

# Medical Insurance Data Science Dashboard

An interactive Streamlit dashboard for exploring medical insurance data, performing statistical hypothesis tests, and predicting medical insurance charges using multiple linear regression.

## Project Overview

This project analyzes the relationship between demographic, lifestyle, and geographical factors and medical insurance charges.

The dashboard contains three sections:

1. **Data Exploration**
   - Dataset preview
   - Descriptive statistics
   - Median and interquartile range
   - Skewness and kurtosis
   - Histograms with KDE
   - BMI versus medical charges scatter plot
   - Correlation matrix

2. **Hypothesis Testing Lab**
   - Shapiro-Wilk normality test
   - Levene's test for equality of variances
   - Independent two-sample t-test or Mann-Whitney U test
   - One-Way ANOVA

3. **Live Prediction & Diagnostics**
   - Multiple linear regression using `statsmodels.api.OLS`
   - Regression coefficients and p-values
   - 95% confidence intervals
   - R-squared and adjusted R-squared
   - Variance Inflation Factor (VIF)
   - Live medical-charge prediction

## Dataset Summary

The project uses the `insurance.csv` dataset.

The dataset contains **1,338 observations** and the following variables:

| Variable | Description | Type |
|---|---|---|
| `age` | Age of the insured person | Numerical |
| `sex` | Sex of the insured person | Categorical |
| `bmi` | Body Mass Index | Numerical |
| `children` | Number of dependent children | Numerical |
| `smoker` | Smoking status | Categorical |
| `region` | Geographical region | Categorical |
| `charges` | Medical insurance charges | Numerical target |

The response variable used in the regression model is `charges`. The predictor variables are `age`, `bmi`, `children`, `smoker`, and `region`.

Categorical variables are converted into numerical dummy variables before fitting the regression model.

## Project Structure

```text
202618023_DS602/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── insurance.csv
└── src/
    ├── __init__.py
    ├── eda.py
    ├── hypothesis.py
    └── modeling.py
```

## Requirements

- Python 3.10 or later
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Streamlit

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Drashti2308/202618023_DS602.git
```

### 2. Open the project directory

```bash
cd 202618023_DS602
```

### 3. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application Locally

Run the Streamlit dashboard using:

```bash
streamlit run app.py
```

Then open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Do not use `python app.py` to launch the dashboard. Use `streamlit run app.py`.

## Statistical Methods

### Descriptive Statistics

The numerical variables are summarized using mean, median, standard deviation, minimum, maximum, quartiles, interquartile range, skewness, and kurtosis.

### Two-Group Hypothesis Testing

The analysis checks assumptions using the Shapiro-Wilk test for normality and Levene's test for equality of variances. Depending on the results, it applies either an independent two-sample t-test or the Mann-Whitney U test.

### One-Way ANOVA

One-Way ANOVA tests whether mean medical charges differ across the four geographical regions.

### Multiple Linear Regression

The regression model is fitted using ordinary least squares:

```python
model = sm.OLS(y, X).fit()
```

The model evaluates the effects of age, BMI, number of children, smoking status, and region on medical insurance charges.

### Regression Diagnostics

The model is assessed using residuals-versus-fitted values, a Q-Q plot, the Jarque-Bera test, and Variance Inflation Factor.

## Concise Statistical Findings

The fitted multiple linear regression model uses **1,338 observations** and explains approximately **75.1% of the variation in medical insurance charges**, with an R-squared value of **0.751**. The adjusted R-squared value is approximately **0.750**, showing that the model retains strong explanatory power after accounting for the number of predictors.

The overall regression model is statistically significant, with an F-statistic of approximately **572.7** and an associated p-value reported as approximately **0.00**. Therefore, the predictors considered together provide statistically significant information about medical insurance charges.

The coefficient table in the dashboard should be used to identify which individual predictors are statistically significant. The sign of each coefficient indicates whether the predictor is associated with an increase or decrease in predicted charges, while the 95% confidence interval indicates the uncertainty around the estimate.

The hypothesis-testing section provides separate conclusions for group comparisons and regional differences. These conclusions should be interpreted using the p-values displayed by the dashboard and a significance level of 0.05.

## Interpretation Guidelines

For hypothesis tests:

```text
p-value <= 0.05:
Reject the null hypothesis.

p-value > 0.05:
Fail to reject the null hypothesis.
```

For regression coefficients:

- A positive coefficient indicates an increase in predicted charges as the predictor increases, holding other variables constant.
- A negative coefficient indicates a decrease in predicted charges as the predictor increases, holding other variables constant.
- A p-value below 0.05 indicates statistical significance at the 5% level.
- If a 95% confidence interval contains zero, the coefficient is not statistically significant at the 5% level.

## Deployment

The dashboard can be deployed using Streamlit Community Cloud.

1. Push the complete project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io).
3. Sign in with GitHub.
4. Select **Deploy an app**.
5. Choose the repository `Drashti2308/202618023_DS602`.
6. Select the `main` branch.
7. Set the main file path to `app.py`.
8. Click **Deploy**.

Make sure `requirements.txt`, `data/insurance.csv`, and the `src` folder are committed to GitHub.

## Important File-Path Note

The application loads the dataset using:

```python
pd.read_csv("data/insurance.csv")
```

Therefore, the dataset must be located at:

```text
data/insurance.csv
```

Do not use a local Windows path because that path will not exist on Streamlit Cloud.

## Limitations

- The regression model describes associations and does not establish causation.
- Medical charges may be influenced by factors not included in the dataset.
- Conclusions depend on the quality and representativeness of the dataset.
- Regression assumptions should be evaluated using both diagnostic plots and formal tests.
- Predictions are model estimates, not guaranteed medical costs.

## Author

**Drashti2308**  
Course/Project: **DS602 - Data Science Laboratory**
