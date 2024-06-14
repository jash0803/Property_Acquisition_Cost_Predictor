import streamlit
import pickle
import pandas as pd
import numpy as np
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

#EXPORATORY DATA ANALYSIS(EDA)
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


# Data to predict on
# 13300000	7420	4	2	3	yes	no	no	no	yes	2	yes	furnished
# 1750000	3850	3	1	2	yes	no	no	no	no	0	no	unfurnished



input_data = (3850, 3, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0)

#changing input data as numpy array
input_data_numpy = np.asarray(input_data)

#reshaping the array because if we don't the model thinks we will provide 543 data but we are provided only 1 and so we need to reshape for one instance.
input_data_numpy_reshape = input_data_numpy.reshape(1,-1)

#standardise the input data
std_data  = scalar.transform(input_data_numpy_reshape)

prediction1 = model1.predict(std_data)
print("Using Linear Regression prediction is : ", prediction1)

prediction2 = model2.predict(std_data)
print("Using Random Forest Regresoor prediction is : ", prediction2)

prediction3 = model3.predict(std_data)
print("Using XGBoost prediction is : ", prediction3)


#Saving the file using Pickle

filename = 'house_price_prediction.sav'
pickle.dump(model2,open(filename,'wb'))
loaded_model = pickle.load(open('house_price_prediction.sav','rb'))

prediction = loaded_model.predict(std_data)

