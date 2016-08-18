#-*- coding: utf-8 -*-
from __future__ import division 

import os,math
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from common import *


def getDateByPerent(start_date,end_date,percent):
    days = (end_date - start_date).days
    target_days = numpy.trunc(days * percent)
    target_date = start_date + timedelta(days=target_days)
    #print days, target_days,target_date
    return target_date

def doPreProcessing(dataset,column_name,threshold=3):
	df_revised = dataset[ dataset[column_name]>0 ]
	return df_revised[((df_revised[column_name] - df_revised[column_name].mean()) / df_revised[column_name].std()).abs() < threshold]


def prepareDataset(dataset,split_ratio=0.75):
	split_index = math.trunc(dataset.shape[0] * 0.75)

	input_column_array = ['Open','Low','Close','Volume']
	output_column = ['High']

	input_data = dataset[input_column_array]
	output_data = dataset[output_column]

	# Create training and test sets
	X_train = input_data[0:split_index]
	Y_train = output_data[0:split_index]

	X_test = input_data[split_index+1:input_data.shape[0]-1]
	Y_test = output_data[split_index+1:input_data.shape[0]-1]

	return X_train,np.ravel(Y_train),X_test,np.ravel(Y_test)


def createSVRModel():
	# Create linear regression object
	regr = SVR(C=1.0, epsilon=0.2, kernel='rbf')
	#regr = RandomForestRegressor()
	return regr

def createRFRModel():
	# Create linear regression object
	regr = RandomForestRegressor()
	#regr = RandomForestRegressor()
	return regr

def trainModel(model, train_x,train_y):
	# Train the model using the training sets
	model.fit(train_x, train_y)

def printTrainingResult(model,test_x,test_y):
	# The coefficients
	#print('Coefficients: \n', model.coef_)
	# The mean square error
	print("Residual sum of squares: %.2f" % np.mean((model.predict(test_x) - test_y) ** 2))

	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % model.score(test_x, test_y))


def plotTrainingResult(model,test_x,test_y):
	# Plot outputs
	pred_y = model.predict(test_x)
	plt.plot(pred_y, color='blue', linewidth=3)
	plt.plot(test_y, color='red', linewidth=3)

	plt.show()


g_dataset = load_stock_data('000240.data')
g_dataset = doPreProcessing(g_dataset,'Volume')
#print g_dataset

train_x,train_y,test_x,test_y = prepareDataset(g_dataset)
#print test_y.as_matrix

#g_model = createSVRModel()
g_model = createRFRModel()

trainModel(g_model,train_x,np.ravel(train_y))
printTrainingResult(g_model,test_x,test_y)
plotTrainingResult(g_model,test_x,test_y)



