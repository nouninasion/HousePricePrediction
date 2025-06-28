# HousePricePrediction
# House Price Prediction

This repository contains a Jupyter Notebook (`HousePredictionPrice.ipynb`) that demonstrates a house price prediction model. The project involves data loading, preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

  - [Project Overview](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#project-overview)
  - [Dataset](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#dataset)
  - [Features](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#features)
  - [Preprocessing Steps](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#preprocessing-steps)
  - [Exploratory Data Analysis (EDA)](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#exploratory-data-analysis-eda)
  - [Model Training](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#model-training)
  - [Model Performance](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#model-performance)
  - [Getting Started](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#getting-started)
  - [Libraries Used](https://github.com/nouninasion/HousePricePrediction/blob/main/README.md#libraries-used)

## Project Overview

The main goal of this project is to predict house prices based on various features such as the number of bedrooms, space, rooms, lot size, tax, bathrooms, garage, and condition. Two regression models, Linear Regression and RandomForestRegressor, are trained and evaluated to achieve this.

## Dataset

The dataset used for this prediction is `realest.csv`. It includes various attributes of houses that influence their price.
im download from this website:https://www.kaggle.com/datasets/tawfikelmetwally/chicago-house-price

## Features

The dataset contains the following features:

  - `Price`: House price (target variable).
  - `Bedroom`: Number of bedrooms.
  - `Space`: Space of the house.
  - `Room`: Number of rooms.
  - `Lot`: Lot size.
  - `Tax`: Property tax.
  - `Bathroom`: Number of bathrooms.
  - `Garage`: Number of garages.
  - `Condition`: Condition of the house (likely a categorical or binary variable).

## Preprocessing Steps

The following preprocessing steps were performed on the data:

  - **Handling Missing Values**: Rows with missing values in `Tax`, `Space`, and `Lot` columns were removed.
  - **Feature Transformation**:
      - A log transformation was applied to the `Space` column to handle skewed data distribution.
  - **Feature Engineering**:
      - `bedroom_ratio`: Calculated as `Bedroom / Room`.
      - `bathroom_ratio`: Calculated as `Bathroom / Room`.
      - `garage_ratio`: Calculated as `Garage / Room`.
  - **Feature Scaling**: `StandardScaler` from `sklearn.preprocessing` was used to scale the features for model training.

## Exploratory Data Analysis (EDA)

Histograms were used to visualize the distribution of each feature, and a heatmap was generated to show the correlation between different features in the training data. This helps in understanding the relationships within the dataset and identifying potential multicollinearity.

## Model Training

Two different regression models were trained for house price prediction:

1.  **Linear Regression**: A basic linear model to establish a baseline.
2.  **RandomForestRegressor**: An ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.

The data was split into training and testing sets using `train_test_split` from `sklearn.model_selection` with a test size of 20%.

## Model Performance

The performance of the models was evaluated using the R-squared score on the test set:

  - **Linear Regression**: R-squared score of 0.687.
  - **RandomForestRegressor**: R-squared score of 0.911.

The RandomForestRegressor model significantly outperforms Linear Regression, indicating its better ability to capture the underlying patterns in the data for house price prediction.

## Getting Started

To run this notebook, you will need Jupyter Notebook or Google Colab and the following libraries installed:

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install these using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Once the dependencies are installed, you can open and run the `HousePredictionPrice.ipynb` file in your Jupyter environment.

## Libraries Used

  - `seaborn`
  - `numpy`
  - `matplotlib.pyplot`
  - `pandas`
  - `sklearn.model_selection`
  - `sklearn.linear_model`
  - `sklearn.preprocessing`
  - `sklearn.ensemble`
  - `sklearn.tree`
