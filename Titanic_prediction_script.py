# -*- coding: utf-8 -*-
"""
Title: Titanic_prediction_script
Author: Faiyaz Hasan
File created on Sept. 3, 2016.

In this script, we predict the survivability in the Titanic disaster. This is 
part of a kaggle competition.

1) We load the test and training data sets from the kaggle website.
2) Preprocessing of the data:
    * Remove or impute NA values.
    * Replace ordinal and nominal features with numerical values.
    * Reshape data sets.
3) Exploratory data analysis: What does the data look like? Form intuition! 
4) Train a machine learning algorithm.    
5) Test the algorithm.
6) Try other algorithms and ways to improve them.

"""

#######################################
# 1. LOAD TRAINING AND TEST DATA SETS #
#######################################

# csv data file locations
testlocation = r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Data\test.csv'
trainlocation = r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Data\train.csv'
testvalueslocation = r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Data\gendermodel.csv'

# import read_csv function from pandas module
import pandas as pd
from pandas import read_csv

# read in data
testdf = pd.read_csv(testlocation)
traindf = pd.read_csv(trainlocation)
testsurvivaldf = pd.read_csv(testvalueslocation)

# glance at  the headings of the data sets
print(traindf.head())
print(testdf.head())
print(testsurvivaldf.head())

################################################################
# 2. Data preprocessing - dimensions and features of datasets. #
################################################################

# find the dimension of the training and test data sets. Find the train-test 
# dataset split ratio.
traindim = traindf.shape
testdim = testdf.shape
testsurvivaldim = testsurvivaldf.shape
print('Training data dimension: ', traindim)
print('Test data set dimension: ', testdim)
print('Test data survival dim.: ', testsurvivaldim)
print('Fraction of training data: ', traindim[0] / (testdim[0] + traindim[0]))
print('Fraction of test data set: ', testdim[0] / (testdim[0] + traindim[0]))

# column names, i.e. features based on which survival is given. Change all
# column names to lower case first for test, train and testsurvival datasets.
traindf.columns = [x.lower() for x in traindf.columns]
testdf.columns = [x.lower() for x in testdf.columns]
testsurvivaldf.columns = [x.lower() for x in testsurvivaldf.columns]

print(traindf.columns)

# check for missing values per column of the training and test data set
print(traindf.isnull().sum())
print(testdf.isnull().sum())
print(testsurvivaldf.isnull().sum())







