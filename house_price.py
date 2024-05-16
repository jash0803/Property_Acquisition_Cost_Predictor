
import streamlit
import pickle
import pandas as pd
import numpy as np
#EXPORATORY DATA ANALYSIS(EDA)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn import metrics


dataframe = pd.read_csv('E:\Machine Learning\HOUSE_PRICE_PREDICTION\Housing.csv')

# printing first 5 columns
print(dataframe.head())

# printing the dimensions of the dataframe
print(dataframe.shape)

# Checking for missing values
print(dataframe.isnull().sum())

#FEATURE ENGINEERING
label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder(categories=[['unfurnished','semi-furnished','furnished']])
dataframe['mainroad'] = label_encoder.fit_transform(dataframe['mainroad'])
dataframe['guestroom'] = label_encoder.fit_transform(dataframe['guestroom'])
dataframe['basement'] = label_encoder.fit_transform(dataframe['basement'])
dataframe['hotwaterheating'] = label_encoder.fit_transform(dataframe['hotwaterheating'])
dataframe['airconditioning'] = label_encoder.fit_transform(dataframe['airconditioning'])
dataframe['prefarea'] = label_encoder.fit_transform(dataframe['prefarea'])
dataframe['furnishingstatus'] = ordinal_encoder.fit_transform(dataframe[['furnishingstatus']])

#Splitting training & testing data
X = dataframe.drop(['price'],axis=1)
Y = dataframe["price"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=0)

#FEATURE SCALING
scalar = StandardScaler()
scalar.fit(train_x)
train_x = scalar.transform(train_x)
test_x = scalar.transform(test_x)
print(train_x, train_y)

#LINEAR REGRESSION
model1 = LinearRegression()
model1.fit(train_x, train_y)
predict_y_model1 = model1.predict(train_x)

# R squared error for Linear Regression
score_model1_1 = metrics.r2_score(train_y, predict_y_model1)
print("R squared error for Linear Regression: ", score_model1_1)

# Mean Absolute Error for Linear Regression
score_model1_2 = metrics.mean_absolute_error(train_y, predict_y_model1)
print('Mean Absolute Error for Linear Regression: ', score_model1_2)


#RANDOM FOREST REGRESSOR
model2 = RandomForestRegressor(n_estimators=20, random_state=8)
model2.fit(train_x, train_y)
predict_y_model2 = model2.predict(train_x)

# R squared error for RANDOM FOREST REGRESSOR
score_model2_1 = metrics.r2_score(train_y, predict_y_model2)
print("R squared error for RANDOM FOREST REGRESSOR: ", score_model2_1)

# Mean Absolute Error for RANDOM FOREST REGRESSOR
score_model2_2 = metrics.mean_absolute_error(train_y, predict_y_model2)
print('Mean Absolute Error for RANDOM FOREST REGRESSOR: ', score_model2_2)


#XGBOOST
model3 = XGBRegressor()
model3.fit(train_x, train_y)
predict_y_model3 = model3.predict(train_x)

# R squared error for XGBOOST
score_model3_1 = metrics.r2_score(train_y, predict_y_model3)
print("R squared error for XGBOOST: ", score_model3_1)

# Mean Absolute Error for XGBOOST
score_model3_2 = metrics.mean_absolute_error(train_y, predict_y_model3)
print('Mean Absolute Error for XGBOOST: ', score_model3_2)


## Data to predict on
data_to_predict = [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 0]]

prediction1 = model1.predict(data_to_predict)
print("Using Linear Regression prediction is : ", prediction1)
prediction2 = model2.predict(data_to_predict)
print("Using Random Forest Regressor prediction is : ", prediction2)
prediction3 = model3.predict(data_to_predict)
print("Using XGBoost prediction is : ", prediction3)