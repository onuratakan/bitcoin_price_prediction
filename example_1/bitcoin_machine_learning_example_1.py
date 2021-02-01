# -*- coding: utf-8 -*-
"""Bitcoin machine learning.ipynb


## Importing the requirements
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn import preprocessing

"""## Uploading the data and setting"""

#Download(https://www.kaggle.com/prasoonkottarathil/btcinusd)  dataset

"""## Working on the data set

Storing the data into a variable
"""

df = pd.read_csv("BTCUSD_day.csv")

#Show the first 7 rows of data
df.head(7)

"""Remove the some data from column"""

df.drop(["Date","Symbol","Open","High","Low","Volume USD","Volume BTC"],1,inplace=True)

"""Reshow first 7 of new datashet"""

df.head(7)

"""A variable for predicting "n" days out into the future"""

prediction_days = 30

"""Create another column shifted "n" units up"""

df["Prediction"] = df[["Close"]].shift(-prediction_days)

"""Reshow first 7 of new datashet"""

df.head(7)

"""Show last 7 of new data set"""

df.tail(7)

"""Create the independent data set"""

# Convert the dataframe to a numpy array and drop the prediction column
X = np.array(df.drop(["Prediction"],1))

#Remove the last "n" rows where "n" is the prediction_days
X = X[:len(df)-prediction_days]

print(X)

"""Create the dependent data set"""

#Convert teh dataframe to a numpy array



y = np.array(df["Prediction"])

#Get all of the values except the last "n" rows
y = y[:-prediction_days]



print(y)

"""Split the data into 60% training and %40 testing"""

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)

"""Set the prediction_days_array to the last 30 rows from the original data set"""

prediction_days_array = np.array(df.drop(["Prediction"],1))[-prediction_days:]

print(prediction_days_array)

"""Create and train the Support Vector Machine (Regression) using radial basic function"""

svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.00001)
svr_rbf.fit(x_train, y_train)

"""Test the model"""

svr_rbf_confidence = svr_rbf.score(x_test, y_test)

print("svr_rbf accuracy: ", svr_rbf_confidence)

"""Print the predicted values"""

svm_prediction = svr_rbf.predict(x_test)

print(svm_prediction)


print()


#Print the actual values
print(y_test)

"""Print the model predictions for the next "n=30" days"""

svm_prediction = svr_rbf.predict(prediction_days_array)
#svm_prediction = svr_rbf.predict(np.array([[36075], [35889], [30809], [31600]]))

print(svm_prediction)

print()

#Print the actual price for Bitcoin for the last 30 days
print(df.tail(prediction_days))
