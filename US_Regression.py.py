# Anthony Rodriguez
# US Housing Linear Regression Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# import data
USA_housing = pd.read_csv('C:/Users/antho/OneDrive/Desktop/Jobs/USA_Housing.csv')

# review the data
USA_housing.head()

USA_housing.info()

USA_housing.describe()

USA_housing.columns

#Drop the address varible (non integer) will have no effect on the validity of the remainder of the data's estimates
#Plot the data for visualizations
sns.pairplot(USA_housing)
plt.show()
sns.displot(USA_housing['Price'])
plt.show()
sns.heatmap(USA_housing.drop('Address', axis=1).corr(), annot=True)
plt.show()

#Train the model
#X= data we are fitting
x = USA_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
#Y= The array with the target varible "Price"
y = USA_housing['Price']

#Splitting the data for training and testing todo get help with this
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.8, random_state=100)

#Fit the data to the training set and evaluate performace to the test set todo get help
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#Using Cross Validation to be the performace metric on training the data using 10 folds (subsets of the data)
def cross_val(linModel):
    pred = cross_val_score(linModel, x, y, cv=10)
    return pred.mean()

#Getting the mean percentage error as an absolute value
def mape_score(y_true, y_pred):
    return np.mean(np.absolute((y_pred - y_true)/y_true)) * 100

#(Print evaluate) and ( return evaluate) for a more observational user oriented analysis
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_percentage_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    mape = mape_score(true, predicted)
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('MAPE:', mape)
    print('R2 Square:', r2_square)
    print('_________________________')

#Getting the errors, re evaluate the metrics for storage and manipulation in a tuple
def evaluate(true, predicted):
    mae = metrics.mean_absolute_percentage_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    mape = mape_score(true, predicted)
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, mape, r2_square

#TODO FINISH THIS watch videos under watch later yt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

x_train = pipeline.fit_transform(x_train)
x_test = pipeline.transform(x_test)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

#Find integer
print(lin_reg.intercept_)

coeff_df = pd.DataFrame(lin_reg.coef_,x.columns, columns=['Coefficient'])
coeff_df

pred = lin_reg.predict(x_test)
plt.scatter(y_test, pred)
plt.show()
sns.distplot((y_test - pred), bins=50)
plt.show()

test_pred = lin_reg.predict(x_test)
train_pred = lin_reg.predict(x_train)
print('Test set evaluation:\n___________________________________')
print_evaluate(y_test,test_pred)
print('Train set evaluation:\n__________________________________')
print_evaluate(y_train, train_pred)

res_df = pd.DataFrame(data=[["Liniar Regression", *evaluate(y_test, test_pred), cross_val(LinearRegression())]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2 Square', "Cross Validation"])
res_df


