# Neural Network for California Housing Price Prediction

## Project Overview

This project implements a neural network from scratch to predict housing prices in California. The assignment is part of an Artificial Intelligence course, focusing on understanding neural network architecture, preprocessing, training, and evaluation.

## Dataset: California Housing Prices

The California Housing Prices dataset contains information about housing in California, designed for statistical analysis and machine learning applications. This dataset is useful for predicting house prices, analyzing market trends, and regression modeling.

### Features

The dataset includes the following variables:
- **longitude**: Longitude coordinate
- **latitude**: Latitude coordinate
- **housing_median_age**: Median age of houses
- **total_rooms**: Total number of rooms
- **total_bedrooms**: Total number of bedrooms
- **population**: Population in the area
- **households**: Number of households
- **median_income**: Median income of residents
- **median_house_value**: Median house value (target variable)
- **ocean_proximity**: Proximity to ocean (categorical)

## Project Tasks

### 1. Data Preprocessing (25 points)

Implemented essential preprocessing steps:

#### Missing Values Strategy
Implemented a strategy to handle missing values in the dataset to ensure data integrity.

#### Outlier Removal
Removed outliers from the target column (median_house_value/price) to improve model performance and prevent skewed predictions.

#### Categorical Feature Encoding
Converted categorical features (ocean_proximity) into numerical representations suitable for neural network training.

**Note**: Dataset was split into training and evaluation sets before preprocessing.

### 2. Simple Neural Network Implementation (15 points)

Implemented a basic neural network with the following architecture:

| NN Module | Parameters |
|-----------|------------|
| Linear | In = input_size, Out = 8 |
| ReLU (Activation Function) | - |
| Linear | In = 8, Out = 1 |

**Total Parameters**: Calculated and reported in the notebook.

### 3. Training the Simple Network (25 points)

#### Hyperparameters

| Hyper Parameters | Value |
|-----------------|-------|
| Batch Size | 64 |
| Train/Validation/Test Ratio | 80% / 10% / 10% |
| Optimizer | Adam |
| Learning Rate | 0.1 |
| Loss Function | MSE |
| Epochs | 60 |

#### Evaluation Metrics

Implemented and calculated the following metrics from scratch:

- **R² (R-squared)**: Measures the proportion of variance in the dependent variable predictable from the independent variables
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, providing error in the same units as the target

**Results**: All metrics and loss curves are reported in the notebook.

### 4. Complex Neural Network Design (20 points)

Designed and implemented a more complex neural network architecture that achieves better performance than the simple network with the same number of epochs.

**Performance Goal**: The improved network demonstrates better metrics across all evaluation criteria.

**Total Parameters**: Calculated and reported in the notebook.

### 5. Comparison and Analysis (15 points)

#### Comparative Analysis

Compared both networks across:
- Hyperparameters
- Training curves
- Evaluation metrics

Provided detailed conclusions and insights from the comparison.

#### Visualizations

- **Loss Curves**: Training and validation loss over epochs for both networks
- **Predicted vs Actual Plot**: Scatter plot showing model predictions against actual values for the complex network

## Performance Requirements

- **Simple Network**: R² score must reach at least **0.60** for full credit on Task 3
- **Complex Network**: R² score must reach at least **0.70** for full credit on Task 4

## Key Implementation Notes

✅ All metrics (R², MAE, MSE, RMSE) implemented from scratch (no sklearn metrics)

✅ Loss curves plotted for both training and validation sets

✅ Metrics reported for all three sets: training, validation, and test

✅ Comprehensive analysis and visualization included

## Files in This Repository

- `2ndQuestion.ipynb`: Complete implementation with all tasks
- `housing.csv`: California Housing Prices dataset
- `README.md`: This file

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- PyTorch / TensorFlow (for neural network implementation)
- Scikit-learn (for preprocessing only)

## How to Run

1. Clone this repository
2. Install required dependencies
3. Open `2ndQuestion.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells sequentially

## Author

Mobin Ghorbani  
CS Student  
University Assignment - Artificial Intelligence Course

## License

This project is for educational purposes as part of university coursework.
