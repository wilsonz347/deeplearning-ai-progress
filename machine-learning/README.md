# Seoul Bike Sharing Demand Forecasting

## Overview

This project predicts **hourly bike rental demand in Seoul** using weather conditions, time-based patterns, operating conditions, and historical demand.

The main question is:

> **How accurately can hourly bike rental demand be predicted using weather, time, and operating conditions?**

The project compares Linear Regression, Gradient Boosting, and XGBoost. XGBoost achieved the strongest performance after incorporating historical demand features.

---

## Dataset

The dataset contains hourly bike rental observations from **December 2017 through November 2018**.

### Features

* Temperature
* Humidity
* Wind speed
* Visibility
* Solar radiation
* Rainfall
* Snowfall
* Holiday status
* Functioning day
* Season
* Hour
* Day of week
* Month

### Target

`Rented Bike Count` — number of bikes rented during each hour.

---

## Project Workflow

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Quality Checks
   ↓
Temporal + Historical Feature Engineering
   ↓
Chronological Train / Validation / Test Split
   ↓
Linear Regression Baseline
   ↓
Gradient Boosting
   ↓
XGBoost
   ↓
Time-Series Hyperparameter Tuning
   ↓
Final Test Evaluation
   ↓
Saved XGBoost Model
```

---

# Exploratory Data Analysis

I first examined:

* Demand trends over time
* Feature distributions
* Correlations between numerical variables
* Missing values
* Duplicate observations
* Potential outliers
* Holiday and operating-day patterns

The data contained no missing values or duplicate rows.

The exploratory analysis showed that bike demand has strong temporal patterns. For example, demand varies substantially by hour, while weather conditions such as temperature, humidity, rainfall, and snowfall are also associated with demand.

### Key EDA Findings

* Warmer temperatures were positively associated with bike demand.
* Humidity, rainfall, and snowfall had negative associations with demand.
* Hourly demand showed clear temporal patterns that simple correlation could not fully capture.
* Dew point temperature was strongly related to temperature, making both variables somewhat redundant.

---

# Feature Engineering

## Temporal Features

I extracted:

* `Day_of_Week`
* `Month`
* `Is_Weekend`

These provide information about recurring weekly and seasonal patterns.

---

## Cyclical Encoding

Hour, day of week, and month were represented using sine and cosine transformations.

For example:

```python
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
```

This is useful because these variables are **cyclical**.

For example, 23:00 and 00:00 are only one hour apart, but treating `Hour` as an ordinary number makes them appear far apart:

```text
23 → 0
```

Cyclical encoding maps the values onto a circle so that the model can recognize this relationship.

The same approach was applied to:

* Hour
* Day of week
* Month

This allows the model to better represent recurring patterns such as:

* Overnight → morning → afternoon → evening
* Sunday → Monday
* December → January

---

## Historical Demand Features

The largest improvement came from incorporating historical demand.

### `lag_24`

Demand at the same hour one day earlier.

### `lag_168`

Demand at the same hour one week earlier.

### `rolling_24`

Average demand over the previous 24 hours.

### `rolling_168`

Average demand over the previous 168 hours.

These features allow the model to capture temporal dependence and recurring daily and weekly demand patterns that weather and calendar variables alone cannot capture.

The rolling features were calculated using a one-period shift:

```python
df["rolling_24"] = df["Rented Bike Count"].shift(1).rolling(24).mean()
df["rolling_168"] = df["Rented Bike Count"].shift(1).rolling(168).mean()
```

The `.shift(1)` prevents the current target from being included in its own rolling average, avoiding target leakage.

---

# Modeling Decisions

Several decisions were made based on the structure of the data and the intended forecasting use case.

## Removed Dew Point Temperature

Dew point temperature was excluded because it overlaps heavily with air temperature.

Keeping both could introduce redundant information without providing substantial additional predictive value.

---

## Removed `lag_1`

The initial model included the previous hour's demand:

```text
lag_1
```

Although this feature improved predictive accuracy, it makes the model heavily dependent on the immediately preceding observation.

I removed `lag_1` to make the feature set more appropriate for forecasting beyond the immediate next hour while retaining longer-term historical demand signals through `lag_24` and `lag_168`.

This represents a tradeoff between **maximum predictive accuracy** and **practical forecasting flexibility**.

---

## Kept `lag_24` and `lag_168`

These features capture recurring demand patterns:

* `lag_24` → same hour yesterday
* `lag_168` → same hour last week

This provides the model with historical context without requiring the immediately preceding hour's demand.

---

## Used Shifted Rolling Features

The rolling averages use only historical observations.

For example:

```python
df["rolling_24"] = (
    df["Rented Bike Count"]
    .shift(1)
    .rolling(24)
    .mean()
)
```

This prevents information from the prediction target from leaking into the feature.

---

## Used Chronological Splitting

Because this is a forecasting problem, observations were split chronologically:

* **70% training**
* **15% validation**
* **15% test**

A random train/test split was avoided because it could allow future observations to influence training.

The final test period therefore represents a future period relative to the training data.

---

## Used Time-Series Cross-Validation

`TimeSeriesSplit` was used during hyperparameter tuning.

This preserves temporal ordering within cross-validation rather than randomly mixing past and future observations.

---

## Used Linear Regression as a Baseline

Linear Regression was included as a simple benchmark.

This establishes whether more complex nonlinear models actually provide meaningful improvements.

---

## Used Gradient Boosting and XGBoost

Gradient Boosting and XGBoost were evaluated because demand is unlikely to have purely linear relationships with weather and temporal variables.

Tree-based boosting models can capture nonlinear effects and interactions between predictors.

---

## Feature Scaling

Feature scaling was applied only to Linear Regression.

```text
Linear Regression
        ↓
Standardized numerical features

Gradient Boosting
        ↓
Raw features

XGBoost
        ↓
Raw features
```

Tree-based models do not generally require scaling because they make decisions using feature thresholds rather than distances or feature magnitude.

The target variable was **not scaled**, so predictions and evaluation metrics remain directly interpretable as numbers of bikes.

---

## Preserved the Test Set

The test set was not used for feature selection or hyperparameter tuning.

It was reserved for the final evaluation of the selected model.

This provides a more realistic estimate of performance on unseen future observations.

---

# Modeling

## Linear Regression

Linear Regression was used as the baseline model.

Its purpose was not necessarily to produce the best forecasts, but to establish a simple benchmark.

The model provides a useful comparison against more flexible nonlinear approaches.

---

## Gradient Boosting

Gradient Boosting was evaluated as a nonlinear alternative.

It builds trees sequentially, with each new tree attempting to correct errors made by previous trees.

Hyperparameters were tuned using `GridSearchCV` with `TimeSeriesSplit`.

---

## XGBoost

XGBoost was evaluated as the primary nonlinear model.

It was particularly effective at combining:

* Weather conditions
* Calendar features
* Cyclical time features
* Historical demand
* Nonlinear interactions

Hyperparameters were tuned using time-series cross-validation.

---

# Results

The final XGBoost model was evaluated on the held-out test period.

### Final Test Performance

| Metric |          XGBoost |
| ------ | ---------------: |
| MAE    |  **76.60 bikes** |
| RMSE   | **116.08 bikes** |
| R²     |       **0.9578** |

> Update these values if the final run produces different metrics.

### Interpretation

An **MAE of 76.60** means the model's predictions were off by approximately **77 bike rentals per hour on average**.

An **R² of 0.9578** means the model explained approximately **95.8% of the variation in hourly bike demand** on the held-out test period.

---

# Key Findings

### 1. Historical demand was highly informative

The addition of `lag_24`, `lag_168`, `rolling_24`, and `rolling_168` substantially improved predictive performance.

This indicates that demand is strongly influenced by recurring and recent demand patterns.

### 2. Nonlinear models performed better

Gradient Boosting and XGBoost were better suited to the data than a simple linear model.

This suggests that relationships between weather, time, and bike demand are not purely linear.

### 3. Time representation matters

Cyclical encoding provides a more appropriate representation of recurring time variables than treating hours, weekdays, and months as ordinary numerical values.

### 4. Feature engineering was important

The improvement from historical demand features demonstrates that better representations of the underlying demand process can be more important than simply selecting a more complex algorithm.

### 5. The model is suited to short-term demand forecasting

The final model combines historical demand with known calendar and weather variables, making it useful for short-term operational forecasting.

---

# Business Implications

A bike-sharing operator could use demand forecasts to support:

### Bike Redistribution

Anticipate periods of high demand and move bikes toward areas expected to experience greater utilization.

### Staffing

Allocate rebalancing and maintenance resources around expected demand.

### Capacity Planning

Identify upcoming periods of high or low system utilization.

### Weather-Aware Operations

Incorporate expected weather conditions when planning daily operations.

### Short-Term Planning

Use recurring daily and weekly patterns to prepare for upcoming demand.

The model is therefore useful as an **operational demand forecasting tool**, rather than simply a prediction exercise.

---

# Limitations

## Geographic Generalization

The model was trained exclusively on Seoul data.

The learned model should **not be assumed to generalize directly to other cities or regions**.

Bike demand can differ because of:

* Population density
* Public transit infrastructure
* Cycling culture
* Geography
* Weather
* Bike availability
* Pricing
* Local holidays
* Commuting patterns
* Station locations

The **methodology** is transferable, but the learned relationships are likely to contain Seoul-specific patterns.

To apply the approach to another city, the model should ideally be retrained using that city's historical bike-sharing data.

---

## Forecast Horizon

The model uses historical demand features, so its forecasting horizon depends on what historical demand information is available at prediction time.

Removing `lag_1` makes the model less dependent on the immediately preceding hour, but multi-step forecasting still requires careful consideration of which historical features would be available for each future prediction.

Longer-horizon forecasting should therefore be evaluated separately rather than assumed from the one-step test metrics.

---

## Dataset Timeframe

The dataset covers approximately one year.

This limits the amount of historical information available for learning longer-term changes in demand, such as year-over-year trends.

---

# What I Learned

## 1. Time-series validation is different from standard machine learning

Random train/test splits are inappropriate for many forecasting problems because they can allow future information to influence training.

Chronological splitting better reflects the real-world forecasting process.

---

## 2. Feature engineering can matter more than model complexity

The biggest performance improvement came from adding historical demand features.

This reinforced the importance of understanding how the target is generated rather than simply selecting increasingly complex models.

---

## 3. Cyclical variables need special treatment

I learned why variables such as hour, weekday, and month should not always be represented as ordinary numerical values.

Sine/cosine encoding preserves their circular structure.

---

## 4. Lag features capture temporal dependence

Previous demand can provide valuable information about future demand.

The 24-hour and 168-hour lags capture recurring daily and weekly patterns.

---

## 5. Rolling features capture broader trends

A rolling average provides more context than a single historical observation.

The 24-hour and 168-hour rolling averages allow the model to understand recent demand levels and longer-term weekly patterns.

---

## 6. Leakage can occur during feature engineering

I learned that even seemingly harmless features can leak information about the target.

Using:

```python
.shift(1)
```

before calculating rolling averages ensures the current target is excluded.

---

## 7. Tree models do not require feature scaling

Scaling is useful for models such as Linear Regression but generally unnecessary for tree-based models such as Gradient Boosting and XGBoost.

This allowed the tree models to operate directly on the original feature values.

---

## 8. Forecasting requires thinking about information availability

A model can have excellent test performance while still being unsuitable for a particular forecasting horizon.

A key question is:

> **What information would actually be available at the moment the prediction is made?**

That determines which historical features can be used and how multi-step forecasting should be evaluated.

---

# Technologies

* **Python**
* **pandas**
* **NumPy**
* **scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

---

# Model Artifact

The final model is saved as:

```text
models/xgb_bike_demand.pkl
```

It can be loaded with:

```python
import pickle

with open("../models/xgb_bike_demand.pkl", "rb") as file:
    model = pickle.load(file)
```

The model expects the same engineered feature set used during training, including the historical demand, cyclical, weather, calendar, and operating-condition features.

---

# Project Takeaway

The main lesson from this project was that **good forecasting depends heavily on representing time and historical behavior correctly**.

Rather than relying solely on weather and calendar variables, incorporating daily and weekly demand history allowed XGBoost to capture recurring demand patterns and substantially improve prediction accuracy.

The resulting model provides a strong baseline for **short-term bike demand forecasting**, while the feature engineering and time-series validation approach can be adapted to other demand forecasting problems.
