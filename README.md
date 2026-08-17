# Machine Learning & AI Projects

A collection of applied machine learning and AI projects focused on building, evaluating, and understanding data-driven models and systems.

Projects are organized by technical area and document the modeling process, key decisions, evaluation results, and practical limitations.

## Projects

### 1. Machine Learning

Projects involving supervised learning, unsupervised learning, feature engineering, model evaluation, and forecasting.

- [Seoul Bike Sharing Demand Forecasting](./machine-learning/)  
  Hourly bike rental demand forecasting using weather, temporal, operating-condition, and historical demand features.

  **Techniques:** XGBoost, Gradient Boosting, Linear Regression, cyclical encoding, lag features, rolling features, time-series cross-validation

  **Result:** 0.88 R² and 133-bike MAE on held-out test data.

### 2. Deep Learning

Projects exploring neural network architectures and deep learning methods.

*Projects coming soon.*

### 3. PyTorch

Projects focused on implementing and training machine learning models using PyTorch.

*Projects coming soon.*

### 4. Generative AI

Projects involving large language models, retrieval-augmented generation, and AI applications.

*Projects coming soon.*

### 5. Production ML

Projects focused on deploying, automating, and maintaining machine learning systems.

*Projects coming soon.*

## Current Technologies

Technologies currently used across the projects in this repository:

**Languages:** Python, SQL

**Data:** pandas, NumPy

**Machine Learning:** scikit-learn, XGBoost

**Visualization:** Matplotlib, Seaborn

**Development:** Git, Jupyter

Additional technologies will be added as they are used in future projects.

## Project Approach

Projects generally follow a structured workflow:

1. Define the problem and prediction objective
2. Explore and validate the data
3. Engineer features based on the underlying problem
4. Establish baseline models
5. Compare and tune candidate models
6. Evaluate performance on unseen data
7. Analyze model behavior, limitations, and tradeoffs
8. Document the resulting approach

The goal is to emphasize **practical modeling decisions and evaluation**, rather than simply implementing algorithms.

## Repository Structure

```text
.
├── machine-learning/
│   ├── data/
│   ├── models/
│   ├── scripts/
│   ├── notebooks/
│   ├── README.md
│   └── requirements.txt
│
├── deep-learning/
│
├── pytorch/
│
├── generative-ai/
│
└── production-ml/
