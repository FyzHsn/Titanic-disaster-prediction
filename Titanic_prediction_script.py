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

#########################
# 2. DATA PREPROCESSING #
#########################

# import numpy package
import numpy as np

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

# remove inconsequential feature columns from test and training data sets.
# remove - passengerid, name, ticket, fare, cabin, embarked.
traindf.drop(['passengerid', 'name', 'ticket', 'fare', 'cabin', 
              'embarked'], inplace=True, axis=1)
testdf.drop(['passengerid', 'name', 'ticket', 'fare', 'cabin', 
             'embarked'], inplace=True, axis=1)
testsurvivaldf.drop(['passengerid'], inplace=True, axis=1)

# Transform nominal values to binary numerical 0/1 representation
traindf = pd.get_dummies(traindf)
testdf = pd.get_dummies(testdf)

print(testdf.head())

# create training and test data frames with features only (X_train, X_test)
# and data frame with survival values only (y_train, y_test).
y_train = traindf.iloc[:, 0].values
y_test = testsurvivaldf.iloc[:, 0].values
X_train = traindf.iloc[:, 1:].values
X_test = testdf.iloc[:, 0:].values

# Test for missing values
print(np.isnan(X_train[:, 1]).sum())

# The age feature has some missing values. Impute the data using mean. 
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(X_train)
X_train = imr.transform(X_train)

print(X_train)


################################
# 3. EXPLORATORY DATA ANALYSIS #
################################

# What proportion of passengers survived?
n_train = y_train.shape[0]
n_test = y_test.shape[0]

# Only about 38% of people survived. 64% of people perished.
print(y_train.sum()/n_train)
print(y_test.sum()/n_test)

# What percentage of men and women survived? Column 4 - sex_female, Column 5 -
# sex_male.
n_male = sum(X_train[:, 5] == 1)
n_female = sum(X_train[:, 4] == 1) 

print('Number of men: ', n_male)
print('Number of women: ', n_female)

male_survivors = sum(X_train[y_train == 1, 5] == 1)
female_survivors = sum(X_train[y_train == 1, 4] == 1) 

print(male_survivors/n_male)
print(female_survivors/n_female)

# What percentage of each of class 1, 2, 3 survived?
n1 = sum(X_train[:, 0] == 1)
n2 = sum(X_train[:, 0] == 2) 
n3 = sum(X_train[:, 0] == 3)

n1_survivors = sum(X_train[y_train == 1, 0] == 1)
n2_survivors = sum(X_train[y_train == 1, 0] == 2) 
n3_survivors = sum(X_train[y_train == 1, 0] == 3)

print('Class 1 survivor percentage: ', n1_survivors/n1)
print('Class 2 survivor percentage: ', n2_survivors/n2)
print('Class 3 survivor percentage: ', n3_survivors/n3)

# What percentage of men and women survived? - Repeat for test sets
n_male = sum(X_test[:, 5] == 1)
n_female = sum(X_test[:, 4] == 1) 

print('Number of men: ', n_male)
print('Number of women: ', n_female)

male_survivors = sum(X_test[y_test == 1, 5] == 1)
female_survivors = sum(X_test[y_test == 1, 4] == 1) 

print(male_survivors)
print(female_survivors)

# What percentage of each of class 1, 2, 3 survived?
n1 = sum(X_test[:, 0] == 1)
n2 = sum(X_test[:, 0] == 2) 
n3 = sum(X_test[:, 0] == 3)

n1_survivors = sum(X_test[y_test == 1, 0] == 1)
n2_survivors = sum(X_test[y_test == 1, 0] == 2) 
n3_survivors = sum(X_test[y_test == 1, 0] == 3)

print('Class 1 survivor percentage: ', n1_survivors/n1)
print('Class 2 survivor percentage: ', n2_survivors/n2)
print('Class 3 survivor percentage: ', n3_survivors/n3) 


############################################
# 4. FEATURE IMPORTANCE VIA RANDOM FORESTS #
############################################

# Labels of column features
feat_labels = traindf.columns[1:]
print(feat_labels)

# From sklearn.ensemble module import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000,
                                random_state=0,
                                n_jobs=-1)

# Fit training data using random forests algorithm
forest.fit(X_train, y_train)
                                
# Important features via .feature_importances_
importances = forest.feature_importances_

# Sort features from highest to lowest importance and store the indices
indices = np.argsort(importances)[::-1]

# Print sorted importances
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]],
                                       importances[indices[f]]))
                                       
# Plot the relative importances - Picture worth a thousand words
import matplotlib.pyplot as plt
plt.title('Titanic Survival Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()                                       

# Use the sequential backwards selection algorithm.
from SBS import SBS




































