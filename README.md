# Predicting-Urban-Bike-Sharing-Demand-with-ML-Models
---
## Project Description
This project leverages machine learning techniques to predict bike-sharing demand in urban areas. By analyzing historical data from bike-sharing systems, the goal is to create accurate demand forecasts that can optimize bike availability and distribution, contributing to more efficient and sustainable urban mobility solutions.

The dataset includes various features such as weather conditions, time of day, day of the week, and seasonal factors, which are used to build predictive models. Through data exploration, feature engineering, and model evaluation, the project aims to uncover patterns and trends that influence bike-sharing demand.

### Problem the Model Solves:
The primary problem addressed by this project is the inefficient allocation of bikes across bike-sharing stations, leading to surplus or shortage of bikes at key locations and times. This project aims to predict bike demand accurately, allowing for better resource management.

### Main Objectives:
- Analyze key factors (weather, time, seasonality) that influence bike-sharing demand.
- Build a predictive model to forecast bike-sharing demand accurately.
- Optimize bike distribution by identifying high-demand locations and times.
- Improve urban mobility by enhancing bike-sharing system efficiency.

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Usage](#usage)
3. [Data](#data)
4. [Model Architecture](#model-architecture)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)

## Installation Instructions

To get started with this project, follow the steps below.

### Requirements:
- Python 3.x (preferably Python 3.7+)
- Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### Dependencies:
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `scikit-learn` (for machine learning algorithms)
- `xgboost` (for gradient boosting model)
- `matplotlib` (for data visualization)
- `seaborn` (for advanced visualization)
- `jupyter` (optional, for running notebooks)

### Setting Up a Virtual Environment:
1. Create a virtual environment:
```bash
python -m venv env
```

2. Activate the environment:
- On Windows:
```bash
.\env\Scripts\activate
```
- On Mac/Linux:
```bash
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Train the Model**: 
Run the following command in your terminal or Jupyter notebook to start training the model:
```bash
python train_model.py
```

2. **Make Predictions**: 
To make predictions on the test data after training the model:
```bash
python predict.py
```

### Input Data Format:
- The dataset is in CSV format with columns like `datetime`, `season`, `holiday`, `workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`, `casual`, `registered`, and `count`.
  
### Expected Output:
- A trained machine learning model that can predict bike-sharing demand (`count`).
- Evaluation metrics such as R², RMSE, RMSLE, and MAE to assess the performance of the model.

## Data

The dataset used in this project contains hourly bike-sharing demand data. It includes several features such as:
- `datetime` (timestamp)
- `season` (spring, summer, fall, winter)
- `holiday` (whether the day is a holiday or not)
- `workingday` (whether it's a working day)
- `weather` (weather condition)
- `temp` (temperature)
- `atemp` (feels-like temperature)
- `humidity` (relative humidity)
- `windspeed` (wind speed)
- `casual` (non-registered user rentals)
- `registered` (registered user rentals)
- `count` (total number of rentals)

The data can be accessed from the provided source or loaded directly into the project directory. If the data is not available locally, download it from the dataset repository or link below.

**Dataset:** [Bike Sharing Demand Dataset](https://www.kaggle.com/competitions/bike-sharing-demand/data)

**Data Preprocessing**:
- Missing values are handled by imputation techniques.
- Temporal features like `hour`, `day`, `month`, and `year` are extracted from the `datetime` column.
- The dataset is split into training (1-16th day) and validation (17-19th day) based on the `day` column.

## Model Architecture

We use multiple machine learning models to predict bike-sharing demand:

1. **Linear Regression**:
   - A simple baseline model for predicting continuous values.
   - Hyperparameters: None.

2. **Random Forest Regression**:
   - A more complex model that handles feature interactions and non-linear relationships.
   - Hyperparameters: Number of trees (default 100), max depth, etc.

3. **XGBoost**:
   - An advanced gradient boosting algorithm for capturing complex patterns in data.
   - Hyperparameters: Learning rate, max depth, subsample, etc.
   - Grid search and regularization are used to fine-tune the model for optimal performance.

4. **Neural Networks** (optional for deep learning):
   - This model could be used for time-series forecasting or capturing complex patterns over time (LSTM or feedforward networks).
  
The best model in our experiments is **XGBoost** with GridSearchCV for hyperparameter tuning.

## Evaluation Metrics

The model performance is evaluated using the following metrics:

- **R²**: Measures the proportion of variance explained by the model.
- **MAE** (Mean Absolute Error): Average of absolute errors between predicted and actual values.
- **RMSE** (Root Mean Squared Error): Measures the average magnitude of the prediction errors.

### Baseline Comparison:
- A simple Linear Regression model is used as a baseline to compare against more complex models like XGBoost.

## Results

### Model Performance:

| Model                          | Training R² | Validation R² | Training RMSE | Validation RMSE |
|---------------------------------|-------------|---------------|---------------|-----------------|
| **Linear Regression**           | 0.71        | 0.71          | 97.02         | 98.31           |
| **XGBoost (No GridSearchCV)**   | 0.86        | 0.83          | 67.39         | 75.88           |
| **XGBoost (With GridSearchCV)** | 0.98        | 0.92          | 28.48         | 52.10           |
| **XGBoost (With Regularization)** | 0.95      | 0.91          | 41.79         | 55.08           |

- **Best Performing Model**: XGBoost with GridSearchCV achieved the best performance, significantly reducing RMSE and improving R² on both training and validation sets.
- **Second Best**: XGBoost with Regularization offers a slight trade-off but helps in slightly reducing overfitting.
- **Linear Regression**: Performs the worst, highlighting the need for more complex models.

### Key Takeaways:
- XGBoost outperforms simpler models by capturing more complex relationships in the data.
- Hyperparameter tuning using GridSearchCV yields significantly better results, especially in terms of RMSE and R².

---

By successfully applying machine learning techniques to predict bike-sharing demand, this project contributes to optimizing bike-sharing systems and improving urban mobility. The insights from the models can help allocate bikes more effectively, reduce congestion, and enhance the user experience in urban transportation systems.
