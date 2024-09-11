# US Housing Linear Regression Model

This project involves creating and training a linear regression model using a dataset of US housing data. The model will predict housing prices based on various features such as average income, house age, number of rooms, bedrooms, and population in the area.

## Table of Contents
1. [Installation](#installation)
2. [Data Overview](#data-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)

## Installation

To run this project, you need to have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
Data Overview
The dataset used in this project contains the following columns:

Avg. Area Income: The average income of residents in the area.
Avg. Area House Age: The average age of houses in the area.
Avg. Area Number of Rooms: The average number of rooms in houses.
Avg. Area Number of Bedrooms: The average number of bedrooms in houses.
Area Population: The population of the area.
Price: The price of the houses (target variable).
Address: The address of the houses (dropped for analysis).
Exploratory Data Analysis (EDA)
We start by loading the dataset and exploring the basic statistics.

python
Copy code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
USA_housing = pd.read_csv('C:/Users/antho/OneDrive/Desktop/Jobs/USA_Housing.csv')

# Basic data overview
print(USA_housing.head())
print(USA_housing.info())
print(USA_housing.describe())
Next, we visualize relationships between variables using pair plots and correlation heatmaps:

python
Copy code
# Pairplot
sns.pairplot(USA_housing)
plt.show()

# Distribution plot of 'Price'
sns.displot(USA_housing['Price'])
plt.show()

# Correlation heatmap (dropping 'Address')
sns.heatmap(USA_housing.drop('Address', axis=1).corr(), annot=True)
plt.show()
Model Training
For model training, we split the data into training and testing sets and apply a linear regression model.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Splitting data into features (X) and target (y)
X = USA_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = USA_housing['Price']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=100)

# Creating a pipeline for standardization and model fitting
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

# Training the model
pipeline.fit(X_train, y_train)

# Model coefficients
print(f"Intercept: {pipeline.named_steps['regression'].intercept_}")
coefficients = pd.DataFrame(pipeline.named_steps['regression'].coef_, X.columns, columns=['Coefficient'])
print(coefficients)
Model Evaluation
We evaluate the model's performance using metrics like MAE, MSE, RMSE, and R².

python
Copy code
from sklearn import metrics
import numpy as np

# Prediction on test data
y_pred_test = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

# Evaluation functions
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_percentage_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(true, predicted)
    print(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR2: {r2}\n{'-'*30}")

# Evaluate on test set
print("Test Set Evaluation:")
print_evaluate(y_test, y_pred_test)

# Evaluate on train set
print("Train Set Evaluation:")
print_evaluate(y_train, y_pred_train)
Conclusion
In this project, we built and evaluated a linear regression model to predict house prices in the US based on various features. The model was trained using 80% of the data and evaluated using performance metrics such as MSE, RMSE, and R².

Future Work
Add more advanced models to improve accuracy.
Tune hyperparameters and implement regularization techniques.
Use feature selection methods to reduce the dimensionality of the model.
