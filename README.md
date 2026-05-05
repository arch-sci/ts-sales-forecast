# Time Series Forecasting — US Retail Clothing Sales

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch-sci/ts-sales-forecast/blob/main/ts-sales-forecast.ipynb)

Forecasting monthly US clothing retail sales (FRED series MRTSSM448USN, 1992–2025) using classical time series methods, with a focus on model justification, cross-validation, and handling structural breaks.

## Problem

A US clothing retail CFO needs a reliable 12-month revenue outlook to plan inventory and staffing. The series spans 407 monthly observations, exhibits a strong upward trend, stable multiplicative seasonality, and a severe structural break in April–May 2020 (COVID-19 store closures).

## Approach

### Exploratory Analysis
- Multiplicative decomposition — trend strength F_T = 1.00, seasonality strength F_S = 0.91
- Seasonal indices confirm December as peak (+65% above average) and January–February as troughs
- ADF and KPSS stationarity tests → d=1, D=1 required for ARIMA modelling
- ACF/PACF analysis to guide SARIMA order selection

### Models
| Model | Description |
|---|---|
| Naïve Seasonal | Baseline — repeats last observed year |
| ETS(A,Ad,M) | Additive errors, damped trend, multiplicative seasonality |
| SARIMA(3,1,1)(0,1,1)[12] | Selected by auto_arima on log-transformed series |

### Evaluation
- Expanding-window cross-validation (5 folds, 12-month horizon)
- Test set: last 24 months (2023–2025)
- Metrics: RMSE, MAE, MAPE + Ljung-Box residual test + empirical prediction interval coverage

## Results

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Naïve Seasonal | 1821 | 1485 | 5.55% |
| **ETS(A,Ad,M)** | **1527** | **1211** | **4.55%** |
| SARIMA(3,1,1)(0,1,1)[12] | 1893 | 1431 | 5.96% |

ETS(A,Ad,M) was selected as the recommended model — lowest error across all metrics and more robust to the COVID structural break than SARIMA, which was fitted on the log-transformed series and systematically underestimated April sales due to the 2020 distortion.

**Further work:** interpolating the COVID months (Apr–Jun 2020) before fitting SARIMA reduced its MAPE from 5.96% to 4.21%, outperforming ETS — suggesting that a SARIMAX model with a COVID dummy variable would be the strongest approach overall.

## Stack
Python · pandas · numpy · statsmodels · pmdarima · matplotlib

## Data
[FRED — MRTSSM448USN](https://fred.stlouisfed.org/series/MRTSSM448USN) (publicly available, free download)
