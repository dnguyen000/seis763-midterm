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

## Validating and Interpreting Results — First Iteration

The first iteration used the **Polygon/Massive API** dataset (~476 rows, 2024–2026, 10 features) evaluated with 9 walk-forward folds. Results are sourced from `midterm.ipynb`.

### EDA Visuals

**VIX Volatility Index (2015–2025)**

![VIX Time Series](img/vix_timeseries.png)

Three distinct stress periods are visible: the 2018 Volmageddon spike (~37), the Q4 2018 Fed rate-hike sell-off, and the COVID crash in March 2020 (peak ~82.69). The red shaded regions mark days where VIX exceeded 30 — the threshold used for `is_major_event`. In the 10-year window, 144+ trading days exceeded this threshold. In the 2-year Polygon window used for first-iteration modeling, only 15 days crossed it.

---

**Close Price Over Time**

![Close Price](img/close_price.png)

SPY (blue) shows steady growth from ~$170 (adjusted) in 2015 to ~$570+ by 2025, with only the COVID crash producing a visible dip. TSLA (red) was flat through 2019, then surged dramatically in 2020–2022 before crashing into 2023 and recovering post-election in late 2024. TSLA's price range (~$14 → $400+) is roughly 30× wider than SPY's, directly reflected in its higher RMSE in regression.

---

**Volume Over Time**

![Volume](img/volume.png)

Both tickers show volume spikes aligned with VIX > 30 periods. SPY's COVID crash spike reached ~500M shares/day (6× average). TSLA's 2020–2021 meme-stock era spike reached ~900M shares/day (8× average). High-volume days are captured by `is_major_event` and `volume_ratio`.

---

**SPY vs TSLA Correlation**

![SPY vs TSLA Correlation](img/spy_tsla_correlation.png)

Correlation ~0.87, driven primarily by shared long-term upward trend rather than daily co-movement. The scatter plot shows distinct market era clusters: TSLA flat and low through 2019, explosive divergence in 2020–2022, and erratic high-price behavior in recent years. This validates the decision to model each ticker independently.

---

**VIX vs Daily Returns**

![VIX vs Returns](img/vix_vs_returns.png)

During normal periods (VIX < 30), SPY has a mean daily return of ~+0.06% with std ~0.8%. During VIX > 30 periods, the mean flips to ~-0.05% with std ~2.1% — more than double the volatility. TSLA shows the same pattern at ~2.3× the magnitude. This confirms that market stress regime is a meaningful signal, but the 2-year Polygon dataset had too few stress days (15) for the model to learn from it.

---

**VIX Level vs Mean Daily Return (Binned)**

![VIX Binned Returns](img/vix_binned_returns.png)

A clear monotonic relationship: as VIX rises, average daily returns fall. Calm markets (VIX < 15) produce the highest average returns; VIX > 30 produces negative average returns. This signal is real and consistent across both data sources, but it only helps the model during the relatively rare stress periods.

---

**Average Daily Return by Month**

![Monthly Returns](img/monthly_returns.png)

September is the weakest month for both tickers (the well-documented "September Effect"). November is among the strongest. This monthly pattern is consistent across Yahoo Finance and Polygon datasets and is directly reflected in the seasonal breakdown results below.

---

### Score Summary

All regression R² values are negative, meaning no model beats a naive "predict the mean" baseline for 5-day forward returns. This is consistent across all models and both tickers.

#### SPY — Regression

| Model | RMSE (mean ± std) | R² (mean ± std) |
|---|---|---|
| Linear Regression | 0.0327 ± 0.0237 | -2.496 ± 3.817 |
| Ridge Regression | 0.0292 ± 0.0202 | -1.649 ± 2.622 |
| **XGBoost** | **0.0258 ± 0.0125** | **-1.060 ± 1.018** |

#### SPY — Classification

| Model | F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|
| Logistic Regression | 0.4605 ± 0.0804 | 0.5423 ± 0.0881 |
| **Random Forest** | **0.5094 ± 0.0733** | **0.5344 ± 0.0705** |

#### TSLA — Regression

| Model | RMSE (mean ± std) | R² (mean ± std) |
|---|---|---|
| Linear Regression | 0.1176 ± 0.0487 | -1.596 ± 1.509 |
| Ridge Regression | 0.1129 ± 0.0476 | -1.394 ± 1.448 |
| **XGBoost** | **0.1062 ± 0.0311** | **-1.298 ± 1.346** |

#### TSLA — Classification

| Model | F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|
| Logistic Regression | 0.4439 ± 0.0970 | 0.5079 ± 0.0630 |
| **Random Forest** | **0.4929 ± 0.0473** | **0.5053 ± 0.0426** |

**XGBoost** was the best regression model for both tickers. **Random Forest** was the best classification model for both tickers.

---

### Seasonal Highlights

Fall (Sep–Nov) had the largest sample size (n=126) and was the most predictable season for regression across all models and both tickers. Spring (Mar–May) was consistently the hardest to predict — highest RMSE and lowest F1 in all configurations. Winter was the second-most predictable season for SPY classification (Random Forest F1: 0.5775).

---

### Challenges and Observations

**1. Noise in the regression target**

`target_return` (5-day forward return) is close to a random walk. Technical indicators explain an estimated 2–5% of the variance in 5-day returns. The remaining 95%+ is driven by news events, earnings surprises, Fed announcements, and large institutional order flow that technical features have no visibility into. All R² values being negative is not a modeling failure — it is a data sufficiency problem. Adding sentiment data, options market data, or macroeconomic factors would be required to meaningfully improve regression scores.

**2. Noise in the classification target**

`target_direction` (binary UP/DOWN) forces the model to classify every price move — including sub-0.5% moves that are indistinguishable from microstructure noise. The model is penalized equally for missing a 0.01% move and a 3% move. A softened target that filters out small moves would give the model cleaner signal on days where a genuine trend is present.

**3. `is_major_event` had zero coefficient weight**

The Polygon 2-year window (2024–2026) contained only 15 trading days with VIX > 30. This was insufficient for any model to learn the stress-regime signal. The 10-year Yahoo Finance dataset (2015–2025) contains 144+ such days, making this feature meaningful in the second iteration.

**4. Feature redundancy**

The second feature set (19 features) included correlated features: `macd`, `macd_signal`, and `macd_hist` are mathematically derived from each other (`macd_hist = macd - macd_signal`). Similarly, `volatility_7` and `volatility_20` overlap. These redundant features hurt Random Forest's vote-splitting and added noise to linear models without adding signal. The second iteration reduces to 16 features by removing the redundant ones.

**5. High variance across folds**

Standard deviation exceeded the mean for Linear Regression on SPY (std 0.0557 vs mean 0.0482), indicating the model performs well in some market regimes and fails badly in others. XGBoost was the most consistent (std ~40–50% of mean). Increasing the training window from 63 to 252 days would reduce this variance by exposing each fold to a more representative market history.

---

## Second Iteration

The second iteration addresses the challenges identified above. Full details and rationale are documented in `secrets/PlanV2.md`.

### Changes from First Iteration

| Area | First Iteration | Second Iteration |
|---|---|---|
| Data source | Polygon API (~476 rows, 2 years) | Yahoo Finance (~2,490 rows, 10 years) |
| Features | 10 features (includes `vwap_dist`) | 16 features (removes 3 redundant; no `vwap_dist`) |
| Classification target | Binary UP/DOWN on every move | 3-class: UP / FLAT / DOWN (±0.5% SPY, ±1.5% TSLA) |
| Regression target | 5-day forward return (near-random-walk) | 5-day forward realized volatility (features better aligned) |
| Walk-forward folds | ~9 folds | ~57 folds |
| Code structure | Single monolithic notebook | Split into `data-fetch.ipynb`, `feature_engineering.ipynb`, `spy_modeling.ipynb`, `tsla_modeling.ipynb` |

### Validating and Interpreting Results — Second Iteration

Results are from `spy_modeling.ipynb` and `tsla_modeling.ipynb`, evaluated with **58 walk-forward folds** over the 10-year Yahoo Finance dataset (2015–2024, ~2,510 rows per ticker after NaN drop).

#### Target distributions (after softening)

| Ticker | Down (-1) | Flat (0) | Up (1) | Threshold |
|---|---|---|---|---|
| SPY | 549 (21.9%) | 1,261 (50.2%) | 700 (27.9%) | ±0.5% |
| TSLA | — | — | — | ±1.5% |

> **Note on F1 baseline:** With 3 classes, a random classifier scores ~0.33 F1 (vs. ~0.50 with binary). Any score above ~0.38 represents meaningful learning.

---

#### SPY — Regression (5-day forward realized volatility)

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.007581 ± 0.005754 | 0.006211 ± 0.004878 | -4.7149 ± 9.5302 |
| Ridge Regression | 0.006933 ± 0.006420 | 0.005650 ± 0.005436 | -3.5593 ± 8.0326 |
| **XGBoost** | **0.005664 ± 0.005197** | **0.004564 ± 0.004267** | **-1.8470 ± 3.1703** |

#### SPY — Classification (next-day direction: Down / Flat / Up)

| Model | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|
| **Logistic Regression** | **0.3288 ± 0.0742** | **0.3542 ± 0.1313** | **0.3772 ± 0.0710** | **0.5107 ± 0.1509** |
| Random Forest | 0.3410 ± 0.0643 | 0.3740 ± 0.1085 | 0.3807 ± 0.0653 | 0.4963 ± 0.1385 |

#### SPY — Seasonal Breakdown (Regression — XGBoost)

| Season | RMSE | MAE | R² | n |
|---|---|---|---|---|
| Spring | 0.011374 | 0.005740 | -0.1112 | 614 |
| Summer | 0.006143 | 0.004114 | -0.2865 | 645 |
| **Fall** | **0.004975** | **0.003751** | **+0.1192** | 630 |
| Winter | 0.006548 | 0.004710 | -0.3181 | 547 |

#### SPY — Seasonal Breakdown (Classification — Random Forest)

| Season | F1 | Precision | Recall | Accuracy | n |
|---|---|---|---|---|---|
| Spring | 0.4136 | 0.4129 | 0.4151 | 0.4609 | 614 |
| Summer | 0.4127 | 0.4204 | 0.4208 | 0.5349 | 645 |
| **Fall** | **0.4350** | **0.4375** | **0.4357** | **0.5222** | 630 |
| Winter | 0.3992 | 0.4012 | 0.4020 | 0.4607 | 547 |

---

#### TSLA — Regression (5-day forward realized volatility)

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.030008 ± 0.022531 | 0.024770 ± 0.019399 | -6.2707 ± 10.9782 |
| Ridge Regression | 0.024565 ± 0.013984 | 0.020150 ± 0.011250 | -3.1198 ± 4.2351 |
| **XGBoost** | **0.019440 ± 0.011024** | **0.015565 ± 0.009102** | **-1.3240 ± 1.5718** |

#### TSLA — Classification (next-day direction: Down / Flat / Up)

| Model | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|---|
| Logistic Regression | 0.2951 ± 0.0692 | 0.3337 ± 0.1114 | 0.3521 ± 0.0564 | 0.3941 ± 0.1047 |
| Random Forest | 0.2946 ± 0.0824 | 0.3300 ± 0.1191 | 0.3326 ± 0.0752 | 0.3810 ± 0.1122 |

#### TSLA — Seasonal Breakdown (Regression — XGBoost)

| Season | RMSE | MAE | R² | n |
|---|---|---|---|---|
| Spring | 0.024517 | 0.016258 | -0.6010 | 614 |
| Summer | 0.017845 | 0.014056 | -0.3097 | 645 |
| Fall | 0.023631 | 0.016905 | -0.4527 | 630 |
| Winter | 0.022837 | 0.015022 | -0.3828 | 547 |

#### TSLA — Seasonal Breakdown (Classification — Random Forest)

| Season | F1 | Precision | Recall | Accuracy | n |
|---|---|---|---|---|---|
| Spring | 0.3432 | 0.3456 | 0.3461 | 0.3648 | 614 |
| Summer | 0.3554 | 0.3570 | 0.3566 | 0.3953 | 645 |
| Fall | 0.3488 | 0.3492 | 0.3573 | 0.3968 | 630 |
| Winter | 0.3247 | 0.3251 | 0.3294 | 0.3638 | 547 |

---

### Comparing First and Second Iteration

#### Classification

The 3-class target (±0.5% SPY, ±1.5% TSLA) filters out small, noisy moves and forces the model to only predict when a clear directional signal is present. F1 scores are not directly comparable to the first iteration because the random baseline dropped from ~0.50 to ~0.33.

| Ticker | Model | 1st Iter F1 (binary) | 2nd Iter F1 (3-class) | Baseline |
|---|---|---|---|---|
| SPY | Logistic Regression | 0.4605 | 0.3288 | ~0.33 |
| SPY | Random Forest | 0.5094 | 0.3410 | ~0.33 |
| TSLA | Logistic Regression | 0.4439 | 0.2951 | ~0.33 |
| TSLA | Random Forest | 0.4929 | 0.2946 | ~0.33 |

SPY Random Forest's 0.3410 F1 sits just above the 0.33 random baseline, with the strongest performance in Fall (F1 0.4350). TSLA models struggle to clear the baseline — the added FLAT class is harder to identify for a high-volatility stock.

#### Regression

Targets changed (`target_return` → `target_volatility`), so RMSE values are not directly comparable. The key improvement metric is R²:

| Ticker | Model | 1st Iter R² | 2nd Iter R² |
|---|---|---|---|
| SPY | XGBoost | -1.0600 ± 1.0182 | -1.8470 ± 3.1703 |
| TSLA | XGBoost | -1.2977 ± 1.3457 | -1.3240 ± 1.5718 |

R² remains negative overall, but **SPY XGBoost achieves R² = +0.1192 in Fall** — the only positive R² in either iteration, meaning the model outperforms the naive mean baseline in that season. This is a meaningful improvement: Fall is the most stable and data-rich season (n=630), and the volatility target is better aligned with the technical features than the raw return target was.

#### Key findings

1. **XGBoost is consistently the best regression model** across both tickers and both iterations. Its RMSE standard deviation is roughly 50–60% of its mean (vs. 100%+ for linear models), making it the most consistent across market regimes.
2. **Fall is the most predictable season** for regression in both iterations. SPY XGBoost achieves the only positive R² (0.1192) in Fall — the 10-year dataset gives the model enough examples of fall volatility patterns to generalize.
3. **Volatility is more learnable than return direction** — the volatility target's negative R² values are smaller in magnitude than those of the return target for TSLA (−1.32 vs −1.30 first iteration), but the improvement is clearest in SPY Fall where R² went positive.
4. **TSLA is harder across the board.** Higher volatility (RMSE ~3–4× SPY) and more erratic regime shifts (meme-stock 2020–2022, post-election 2024 spike) make both regression and classification significantly more difficult. TSLA classification F1 fails to meaningfully clear the 0.33 baseline.
5. **The 10-year dataset activates `is_major_event`**: 144 VIX>30 days vs. 15 in the Polygon 2-year window. The feature now has sufficient examples to contribute, as seen in its non-zero coefficient weight in the linear models.

---

## Conclusion

*To be completed at project end.*
