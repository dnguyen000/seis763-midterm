# SEIS-763: Midterm Project — Stock Market Movement Prediction

**Team Members:** Brahmee Adhikari, Don C. Nguyen

---

## Goal

This project builds machine learning models to predict stock price movement using historical daily price data. Two tasks are performed:

1. **Classification** — Predict whether a stock's price will go **UP or DOWN** the next trading day.
2. **Regression** — Predict **how much the price will change** over the next 5 trading days.

Both tasks are applied to two tickers:
- **SPY** — An S&P 500 ETF representing a stable, broad-market benchmark
- **TSLA** — Tesla, Inc., a high-volatility individual stock

Comparing the two tickers reveals whether stability or volatility makes a stock easier to predict, and how extreme market events affect model performance.

---

## Data Source

Historical daily stock price data is sourced from the **Polygon.io API** (formerly Massive):
- https://polygon.io

Data spans approximately **2 years** of trading days for both SPY and TSLA. Each trading day includes the following native attributes returned by the API:

| Attribute | Description |
|---|---|
| `open` | Price at market open |
| `high` | Highest price of the day |
| `low` | Lowest price of the day |
| `close` | Price at market close |
| `volume` | Number of shares traded |
| `vwap` | Volume Weighted Average Price |

From these attributes, additional features are derived (see Key Terms below).

---

## Key Terms

### Stock Market Terms

| Term | Definition |
|---|---|
| **Daily Return** | The percentage change in a stock's closing price from one day to the next. Captures short-term momentum. |
| **Weekly Return** | The percentage change in closing price over the past 5 trading days (~1 week). |
| **Moving Average (20-day)** | The average closing price over the past 20 trading days. Smooths out noise and reveals the medium-term trend. |
| **Distance from 20-day MA** | How far today's price is above or below the 20-day moving average, expressed as a percentage. Indicates whether a stock is overbought or oversold relative to its recent trend. |
| **Daily Range** | The difference between a day's high and low price, normalized by the closing price. Measures intraday volatility. |
| **Volume** | The total number of shares traded in a day. High volume often signals strong conviction behind a price move. |
| **Volume Change** | The day-over-day percentage change in trading volume. |
| **VWAP (Volume Weighted Average Price)** | The average price a stock traded at throughout the day, weighted by volume. A common institutional benchmark. |
| **VWAP Distance** | How far the closing price is from VWAP. Indicates whether the stock closed above or below the average trade price. |
| **Volatility (20-day)** | The rolling standard deviation of daily returns over 20 days. Higher values mean more unpredictable price swings. |
| **RSI (Relative Strength Index)** | A momentum indicator scaled 0–100. Values above 70 suggest a stock is overbought; below 30 suggests oversold. Calculated over 14 days. |
| **Bollinger Band Position** | Where today's closing price sits within the Bollinger Bands, which are set 2 standard deviations above and below the 20-day moving average. Values near +1 or -1 suggest price is at the edge of its recent range. |
| **VIX (CBOE Volatility Index)** | A real-time index published by the Chicago Board Options Exchange (CBOE) that measures the market's expectation of volatility over the next 30 days. It is derived from the implied volatility of S&P 500 options — when traders are willing to pay more for options (i.e., hedging against large swings), VIX rises. Often called the "fear gauge," VIX tends to spike sharply during crises and remain elevated until conditions stabilize. A VIX above 30 is a widely used threshold for heightened market stress. In this project, VIX daily closing prices are fetched from Polygon.io (`I:VIX`) and used to derive the extreme event indicator. |
| **Extreme Event Indicator** | A boolean (True/False) flag marking trading days where the VIX closing price exceeded 30, indicating a period of elevated market fear or stress. Rather than hardcoding specific crash dates, this approach is data-driven: it automatically captures any shock period in the dataset where volatility was abnormally high. Helps the model distinguish genuine market disruptions from normal day-to-day noise. |
| **SPY** | The SPDR S&P 500 ETF Trust. Tracks the S&P 500 index, representing 500 large U.S. companies. Used here as a stable market benchmark. |
| **TSLA** | Tesla, Inc. A high-growth, high-volatility individual stock used here as a contrast to SPY. |

### Machine Learning Terms

| Term | Definition |
|---|---|
| **Feature (Independent Variable)** | An input variable used by the model to make a prediction. In this project, features include daily return, RSI, volume change, etc. |
| **Target (Dependent Variable)** | The value the model is trying to predict. Here: next-day direction (classification) or 5-day price change (regression). |
| **Classification** | A type of ML task where the model predicts a discrete category. Here: UP or DOWN. |
| **Regression** | A type of ML task where the model predicts a continuous numeric value. Here: 5-day forward return. |
| **Logistic Regression** | A classification algorithm that estimates the probability of a binary outcome (UP or DOWN) using a logistic curve. Simple and interpretable. |
| **Linear Regression** | A regression algorithm that fits a straight line through training data to predict continuous values. |
| **Ridge Regression** | A variant of linear regression that adds a penalty for large coefficients, reducing overfitting when features are correlated. |
| **Random Forest** | An ensemble of many decision trees that each vote on the prediction. More powerful than a single tree and provides built-in feature importance scores. |
| **XGBoost** | A gradient boosting algorithm that builds trees sequentially, each correcting the errors of the previous. Often the top performer on structured/tabular data. |
| **Training** | The process of fitting a model to historical data so it learns patterns. |
| **Testing** | Evaluating the model on data it has never seen to measure real-world performance. |
| **Overfitting** | When a model learns the training data too precisely, including its noise, and fails to generalize to new data. |
| **Walk-Forward Validation** | A time-aware testing strategy where the model is trained on a past window of data and tested on the immediately following window, repeating across multiple periods. Prevents data leakage and simulates real trading conditions. |
| **Data Leakage** | When information from the future accidentally influences the training process, producing misleadingly optimistic results. |
| **Feature Importance / Weights** | A score assigned to each feature indicating how much it influenced the model's predictions. Computed via `model.coef_` for linear models and `model.feature_importances_` for tree-based models. |
| **Standardization (StandardScaler)** | Rescaling features to have a mean of 0 and standard deviation of 1, so no single feature dominates due to scale differences. |
| **MAE (Mean Absolute Error)** | Average absolute difference between predicted and actual values. Intuitive and robust to outliers. |
| **RMSE (Root Mean Squared Error)** | Square root of the average squared error. Penalizes large mistakes more than MAE — important in finance where big misses are costly. |
| **R² (R-squared)** | Proportion of variance in the target explained by the model. A value of 1.0 is perfect; negative values mean the model performs worse than simply predicting the mean. |
| **Precision** | Of all days the model predicted as UP, the fraction that actually went UP. Low precision means the model generates many false buy signals — a trader acting on every UP prediction would frequently lose money on days that actually went down. |
| **Recall** | Of all days that actually went UP, the fraction the model correctly identified. Low recall means the model misses many genuine opportunities — a conservative model that rarely predicts UP will have high precision but low recall. |
| **F1 Score** | Harmonic mean of precision and recall. A single number that balances both risks: false buy signals (low precision) and missed opportunities (low recall). More informative than accuracy alone when classes are nearly balanced. |
| **Confusion Matrix** | A table showing correct and incorrect predictions broken down by class (e.g., Predicted Up vs. Actual Up). |
| **Seasonal Analysis** | Grouping model results by time of year (spring, summer, fall, winter) to identify whether market conditions in certain seasons are more predictable. |

---

## How Testing Is Done

Testing uses **walk-forward validation** — a time-series aware evaluation strategy that prevents data leakage and simulates realistic trading conditions.

### Window Sizes
- **Training window:** 63 trading days (~3 months)
- **Test window:** 42 trading days (~2 months)
- The 2-year dataset yields approximately **8–9 folds**

### Process
Each fold trains the model exclusively on its 63-day window, then predicts the following 42 days. The next fold begins immediately after the previous test window ends:

```
Fold 1: Train [day 1   → day 63],  Test [day 64  → day 105]
Fold 2: Train [day 64  → day 126], Test [day 127 → day 168]
Fold 3: Train [day 127 → day 189], Test [day 190 → day 231]
...
```

This approach captures performance across multiple market conditions rather than a single arbitrary split.

### Metrics Reported
Results are aggregated as **mean ± standard deviation** across all folds.

| Task | Primary Metric | Supporting Metrics |
|---|---|---|
| Regression | RMSE | MAE, R² |
| Classification | F1 Score | Accuracy, Confusion Matrix |

Standard deviation across folds measures **consistency** — a model with lower std is more reliable across different market conditions.

---

## Models

All models are evaluated using walk-forward validation across both tickers independently.

### Regression Models
Predict the **5-day forward return** — the percentage price change from today's close to the close 5 trading days from now. A positive value means the model expects the price to rise; negative means it expects a decline.

| Model | Purpose | Primary Metric | Supporting Metrics |
|---|---|---|---|
| **Linear Regression** | Baseline model. Fits a straight line through the feature space to predict the 5-day return as a weighted sum of inputs. Coefficients reveal which features most influence the prediction. | RMSE | MAE, R² |
| **Ridge Regression** | Extension of linear regression with L2 regularization — penalizes large coefficients to reduce overfitting. Particularly useful here because technical indicators (RSI, Bollinger Band position, distance from MA) are often correlated with each other. | RMSE | MAE, R² |
| **XGBoost** | Gradient boosting ensemble that builds decision trees sequentially, each correcting the residual errors of the prior tree. Captures non-linear relationships between features and returns that linear models cannot, and provides feature importance scores. | RMSE | MAE, R² |

### Classification Models
Predict the **next-day price direction** — whether the stock will close higher (UP = 1) or lower (DOWN = 0) than today.

| Model | Purpose | Primary Metric | Supporting Metrics |
|---|---|---|---|
| **Logistic Regression** | Baseline classifier. Models the probability of an UP move using a logistic curve. Simple and interpretable — log-odds coefficients show which features linearly push the prediction toward UP or DOWN. | F1 | Precision, Recall, Accuracy, Confusion Matrix |
| **Random Forest** | Ensemble of decision trees that each vote independently on direction. More robust than a single tree; handles non-linear feature interactions and is less sensitive to noisy features. Provides feature importance scores based on how often each feature is used to split. | F1 | Precision, Recall, Accuracy, Confusion Matrix |

---

## Results

Results are reported as **mean ± standard deviation** across all walk-forward folds. Standard deviation measures consistency — a lower value means the model performs reliably across different market conditions, not just in one favorable period.

### SPY — Regression (5-day forward return)

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.032680 ± 0.023691 | 0.026165 ± 0.020355 | -2.4955 ± 3.8170 |
| Ridge Regression | 0.029199 ± 0.020192 | 0.023036 ± 0.017339 | -1.6488 ± 2.6224 |
| XGBoost | 0.025754 ± 0.012481 | 0.019259 ± 0.009262 | -1.0600 ± 1.0182 |

### SPY — Classification (next-day direction)

| Model | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|
| Logistic Regression | 0.4605 ± 0.0804 | 0.5752 ± 0.1335 | 0.5243 ± 0.0469 | 0.5423 ± 0.0881 |
| Random Forest | 0.5094 ± 0.0733 | 0.5613 ± 0.0647 | 0.5505 ± 0.0498 | 0.5344 ± 0.0705 |

### TSLA — Regression (5-day forward return)

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.117630 ± 0.048685 | 0.097380 ± 0.042914 | -1.5955 ± 1.5091 |
| Ridge Regression | 0.112860 ± 0.047628 | 0.093590 ± 0.041686 | -1.3942 ± 1.4484 |
| XGBoost | 0.106195 ± 0.031106 | 0.084940 ± 0.025590 | -1.2977 ± 1.3457 |

### TSLA — Classification (next-day direction)

| Model | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|
| Logistic Regression | 0.4439 ± 0.0970 | 0.4514 ± 0.1323 | 0.5123 ± 0.0390 | 0.5079 ± 0.0630 |
| Random Forest | 0.4929 ± 0.0473 | 0.5222 ± 0.0425 | 0.5185 ± 0.0404 | 0.5053 ± 0.0426 |

### Seasonal Breakdown

Performance broken down by season (Spring: Mar–May, Summer: Jun–Aug, Fall: Sep–Nov, Winter: Dec–Feb) to identify which market periods are most predictable.

#### SPY — Regression

| Model | Season | RMSE | MAE | R² | n |
|---|---|---|---|---|---|
| Linear Regression | Spring | 0.082822 | 0.072579 | -3.7047 | 63 |
| Linear Regression | Summer | 0.025874 | 0.021037 | -1.9116 | 80 |
| Linear Regression | Fall | 0.021934 | 0.015780 | -0.8844 | 126 |
| Linear Regression | Winter | 0.020531 | 0.015106 | -0.7362 | 109 |
| Ridge Regression | Spring | 0.072096 | 0.062195 | -2.5650 | 63 |
| Ridge Regression | Summer | 0.021907 | 0.017346 | -1.0873 | 80 |
| Ridge Regression | Fall | 0.019774 | 0.014445 | -0.5315 | 126 |
| Ridge Regression | Winter | 0.020126 | 0.014509 | -0.6684 | 109 |
| XGBoost | Spring | 0.050827 | 0.039220 | -0.7719 | 63 |
| XGBoost | Summer | 0.024837 | 0.018364 | -1.6830 | 80 |
| XGBoost | Fall | 0.020643 | 0.014256 | -0.6691 | 126 |
| XGBoost | Winter | 0.018489 | 0.014161 | -0.4080 | 109 |

#### SPY — Classification

| Model | Season | F1 | Precision | Recall | Accuracy | n |
|---|---|---|---|---|---|---|
| Logistic Regression | Spring | 0.3930 | 0.3957 | 0.3935 | 0.3968 | 63 |
| Logistic Regression | Summer | 0.4264 | 0.4436 | 0.4648 | 0.5125 | 80 |
| Logistic Regression | Fall | 0.5478 | 0.5713 | 0.5540 | 0.6190 | 126 |
| Logistic Regression | Winter | 0.4871 | 0.6046 | 0.5497 | 0.5596 | 109 |
| Random Forest | Spring | 0.4439 | 0.4574 | 0.4583 | 0.4444 | 63 |
| Random Forest | Summer | 0.4617 | 0.4783 | 0.4789 | 0.4625 | 80 |
| Random Forest | Fall | 0.5553 | 0.5562 | 0.5591 | 0.5714 | 126 |
| Random Forest | Winter | 0.5775 | 0.6077 | 0.5910 | 0.5963 | 109 |

#### TSLA — Regression

| Model | Season | RMSE | MAE | R² | n |
|---|---|---|---|---|---|
| Linear Regression | Spring | 0.181320 | 0.153579 | -1.4741 | 63 |
| Linear Regression | Summer | 0.085474 | 0.065566 | -1.2383 | 80 |
| Linear Regression | Fall | 0.125973 | 0.090276 | -0.9983 | 126 |
| Linear Regression | Winter | 0.112173 | 0.096461 | -1.2922 | 109 |
| Ridge Regression | Spring | 0.170715 | 0.143752 | -1.1932 | 63 |
| Ridge Regression | Summer | 0.079302 | 0.061654 | -0.9267 | 80 |
| Ridge Regression | Fall | 0.123982 | 0.088431 | -0.9356 | 126 |
| Ridge Regression | Winter | 0.109250 | 0.094000 | -1.1743 | 109 |
| XGBoost | Spring | 0.127032 | 0.103329 | -0.2144 | 63 |
| XGBoost | Summer | 0.100555 | 0.079171 | -2.0978 | 80 |
| XGBoost | Fall | 0.115734 | 0.081984 | -0.6866 | 126 |
| XGBoost | Winter | 0.099294 | 0.081961 | -0.7961 | 109 |

#### TSLA — Classification

| Model | Season | F1 | Precision | Recall | Accuracy | n |
|---|---|---|---|---|---|---|
| Logistic Regression | Spring | 0.4061 | 0.4388 | 0.4576 | 0.4444 | 63 |
| Logistic Regression | Summer | 0.4591 | 0.5000 | 0.5000 | 0.5000 | 80 |
| Logistic Regression | Fall | 0.4973 | 0.5078 | 0.5067 | 0.5317 | 126 |
| Logistic Regression | Winter | 0.5080 | 0.5651 | 0.5516 | 0.5229 | 109 |
| Random Forest | Spring | 0.4714 | 0.4808 | 0.4818 | 0.4762 | 63 |
| Random Forest | Summer | 0.4972 | 0.5000 | 0.5000 | 0.5000 | 80 |
| Random Forest | Fall | 0.5453 | 0.5466 | 0.5474 | 0.5476 | 126 |
| Random Forest | Winter | 0.4771 | 0.4839 | 0.4839 | 0.4771 | 109 |

---

## Validating and Interpreting Results

*To be completed after results are available.*

---

## Conclusion

*To be completed at project end.*
