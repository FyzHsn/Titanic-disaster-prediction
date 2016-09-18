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
X_test = imr.transform(X_test)


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
plt.title('Titanic Survival Feature Importance \n using Random Forests')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig(r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Figs\Feat_importance_random_forest.png')
#plt.clf()              
plt.show()

# Standardize data before fitting - I am not sure what it would do to 
# standardize the male female and passenger class data.
# It would have made sense to use the StandardScaler function from the 
# sklearn.preprocessing module but it leads to some issues. So, I will only
# normalize the age data which has a pretty big spread.
X_train_std = X_train
X_test_std = X_test
X_test_std[:, 1] = (X_test[:, 1]-X_test[:, 1].mean())/X_test[:, 1].std() 
X_train_std[:, 1] = (X_train[:, 1]-X_train[:, 1].mean())/X_train[:, 1].std()

# Use the sequential backwards selection algorithm - in conjucntion with k
# nearest neighbors classifier.
from SBS import SBS

# knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)

sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# Plot of feature importances - using knn classifier
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylabel('Prediction accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.title('Sequential Backward Selection using \n knn classifier')
#plt.savefig(r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Figs\Feat_importance_SBS_knn.png')
#plt.clf()   
plt.show()

# Support vector machine - algorithm
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)

sbs = SBS(svm, k_features=1)
sbs.fit(X_train_std, y_train)

# Plot of feature importances - using support vector machine classifier
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylabel('Prediction accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.title('Sequential Backward Selection using \n support vector machine')
#plt.savefig(r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Figs\Feat_importance_SBS_svm.png')
#plt.clf()
plt.show()

# Feature importance using L1 normalization and the Logistic regression
from sklearn.linear_model import LogisticRegression

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=feat_labels[column],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')             
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
ncol=1, fancybox=True)  
#plt.savefig(r'C:\Users\Windows\Dropbox\AllStuff\Titanic_Kaggle\Figs\Feat_importance_l1norm.png', bbox_inches='tight')
#plt.clf()           
plt.show()             

####################################################
# 5. PREDICTIONS USING MACHINE LEARNING ALGORITHMS #
####################################################

# Logistic Regression
print('Logistic Regression')
lr = LogisticRegression(penalty='l1', C=0.1, random_state=0)
lr.fit(X_train_std, y_train)
print('Training accuracy: ', lr.score(X_train_std, y_train))
print('Test accuracy: ', lr.score(X_test_std, y_test))

# Support Vector Machines
print('Support Vector Machines')
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
print('Training accuracy: ', svm.score(X_train_std, y_train))
print('Test accuracy: ', svm.score(X_test_std, y_test))

# Decision Tree Learning
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Decision Tree Learning')
print('Training accuracy: ', tree.score(X_train, y_train))
print('Test accuracy: ', tree.score(X_test, y_test))


# Random Forest Classifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train, y_train)                                
print('Random Forests')
print('Training accuracy: ', forest.score(X_train, y_train))
print('Test accuracy: ', forest.score(X_test, y_test))

# SGD classifier
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(penalty='elasticnet', loss='perceptron', n_iter=100,
                    learning_rate='optimal', random_state=0, alpha=0.001)
ppn.fit(X_train_std, y_train)
print('Perceptron')
print('Training accuracy: ', ppn.score(X_train_std, y_train))
print('Test accuracy: ', ppn.score(X_test_std, y_test))

# Construct table to display test and training results. 
lrtrainscore = lr.score(X_train_std, y_train)
svmtrainscore = svm.score(X_train_std, y_train)
treetrainscore = tree.score(X_train_std, y_train)
foresttrainscore = forest.score(X_train_std, y_train)
ppntrainscore =  ppn.score(X_train_std, y_train)

lrtestscore = lr.score(X_test_std, y_test)
svmtestscore = svm.score(X_test_std, y_test)
treetestscore = tree.score(X_test_std, y_test)
foresttestscore = forest.score(X_test_std, y_test)
ppntestscore =  ppn.score(X_test_std, y_test)

algorithmdf = pd.DataFrame([
                    ['Logistic regression', round(lrtrainscore*100, 2), 
                     round(lrtestscore*100, 2)],
                    ['Support vector machine', round(svmtrainscore*100, 2), 
                     round(svmtestscore*100, 2)],
                    ['Decision tree', round(treetrainscore*100, 2), 
                     round(treetestscore*100, 2)],
                    ['Random Forests', round(foresttrainscore*100, 2), 
                     round(foresttestscore*100, 2)],
                    ['Perceptron', round(ppntrainscore*100, 2), 
                     round(ppntestscore*100, 2)]])   

algorithmdf.columns = ['Algorithm name', 'Train score (%)', 'Test score (%)']
                       
# Construct table of results
from pandas.tools.plotting import table
fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, algorithmdf, loc='upper right', 
              colWidths=[0.21]*len(algorithmdf.columns))  
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
#plt.savefig('performance_table.png')
#plt.clf()
plt.show()

###################################
# 5. PRINCIPAL COMPONENT ANALYSIS #
###################################

# compute the covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

# we plot the variance explained by eigenvalues
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(0, len(eigen_vals)), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(0, len(eigen_vals)), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()         

# applying PCA in scikit-learn
from sklearn.decomposition import PCA
lr = LogisticRegression()
for compnum in range(1, X_train.shape[1]):
    pca = PCA(n_components=compnum)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    print('PCA test score: ', lr.score(X_test_pca, y_test))
    
# variance explained ratio via scikit-learn
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)

###################################
# 6. LINEAR DISCRIMINANT ANALYSIS #
###################################

# linear discriminant analysis (LDA) vs scikit-learn
from sklearn.lda import LDA
lr = LogisticRegression()
for compnum in range(1, X_train.shape[1]):
    lda = LDA(n_components=compnum)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)
    lr.fit(X_train_lda, y_train)
    print('LDA test score: ', lr.score(X_test_lda, y_test))

"""
Unlike PCA, LDA still leads to an improvement in performance despite having
removed misleading variables. Hence, let us apply it to the other algorithms as
well.

"""    
    
# Logistic Regression
print('Logistic Regression')
lr = LogisticRegression(penalty='l1', C=0.1, random_state=0)
lr.fit(X_train_lda, y_train)
print('Training accuracy: ', lr.score(X_train_lda, y_train))
print('Test accuracy: ', lr.score(X_test_lda, y_test))

# Support Vector Machines
print('Support Vector Machines')
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_lda, y_train)
print('Training accuracy: ', svm.score(X_train_lda, y_train))
print('Test accuracy: ', svm.score(X_test_lda, y_test))

# Decision Tree Learning
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train_lda, y_train)

print('Decision Tree Learning')
print('Training accuracy: ', tree.score(X_train_lda, y_train))
print('Test accuracy: ', tree.score(X_test_lda, y_test))


# Random Forest Classifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train_lda, y_train)                                
print('Random Forests')
print('Training accuracy: ', forest.score(X_train_lda, y_train))
print('Test accuracy: ', forest.score(X_test_lda, y_test))

# SGD classifier
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(penalty='elasticnet', loss='perceptron', n_iter=100,
                    learning_rate='optimal', random_state=0, alpha=0.001)
ppn.fit(X_train_lda, y_train)
print('Perceptron')
print('Training accuracy: ', ppn.score(X_train_lda, y_train))
print('Test accuracy: ', ppn.score(X_test_lda, y_test))

# Construct table to display test and training results. 
lrtrainscore = lr.score(X_train_lda, y_train)
svmtrainscore = svm.score(X_train_lda, y_train)
treetrainscore = tree.score(X_train_lda, y_train)
foresttrainscore = forest.score(X_train_lda, y_train)
ppntrainscore =  ppn.score(X_train_lda, y_train)

lrtestscore = lr.score(X_test_lda, y_test)
svmtestscore = svm.score(X_test_lda, y_test)
treetestscore = tree.score(X_test_lda, y_test)
foresttestscore = forest.score(X_test_lda, y_test)
ppntestscore =  ppn.score(X_test_lda, y_test)

algorithmdf = pd.DataFrame([
                    ['Logistic regression', round(lrtrainscore*100, 2), 
                     round(lrtestscore*100, 2)],
                    ['Support vector machine', round(svmtrainscore*100, 2), 
                     round(svmtestscore*100, 2)],
                    ['Decision tree', round(treetrainscore*100, 2), 
                     round(treetestscore*100, 2)],
                    ['Random Forests', round(foresttrainscore*100, 2), 
                     round(foresttestscore*100, 2)],
                    ['Perceptron', round(ppntrainscore*100, 2), 
                     round(ppntestscore*100, 2)]])   

algorithmdf.columns = ['Algorithm name', 'Train score (%)', 'Test score (%)']
                       
# Construct table of results
from pandas.tools.plotting import table
fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, algorithmdf, loc='upper right', 
              colWidths=[0.21]*len(algorithmdf.columns))  
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
#plt.savefig('performance_table_LDA.png')
#plt.clf()
plt.show()    


#########################################
# 7. K-FOLD STRATIFIED CROSS-VALIDATION #
#########################################

# Pipeline + stratified K fold module
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

lr = LogisticRegression(penalty='l1', C=0.1, random_state=0)
lr.fit(X_train_lda, y_train)
lr_scores = cross_val_score(estimator=lr,
                            X=X_train_lda,
                            y=y_train,
                            cv=15)

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_lda, y_train)                         
svm_scores = cross_val_score(estimator=svm,
                             X=X_train_lda,
                             y=y_train,
                             cv=15)
                         
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train_lda, y_train)
tree_scores = cross_val_score(estimator=tree,
                              X=X_train_lda,
                              y=y_train,
                              cv=15)
                         
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train_lda, y_train)   
forest_scores = cross_val_score(estimator=forest,
                                X=X_train_lda,
                                y=y_train,
                                cv=15)
                                            
ppn = SGDClassifier(penalty='elasticnet', loss='perceptron', n_iter=100,
                    learning_rate='optimal', random_state=0, alpha=0.001)
ppn.fit(X_train_lda, y_train)
ppn_scores = cross_val_score(estimator=ppn,
                             X=X_train_lda,
                             y=y_train,
                             cv=15)
                                                 
# table of cross-validated scores
algorithmdf = pd.DataFrame([
                    ['Logistic regression', 
                     round(np.mean(lr_scores)*100, 2),
                     round(np.std(lr_scores)*100, 2)],
                    ['Support vector machine', 
                     round(np.mean(svm_scores)*100, 2),
                     round(np.std(svm_scores)*100, 2)],
                    ['Decision tree', 
                     round(np.mean(tree_scores)*100, 2),
                     round(np.std(tree_scores)*100, 2)],
                    ['Random Forests', 
                     round(np.mean(forest_scores)*100, 2),
                     round(np.std(forest_scores)*100, 2)],
                    ['Perceptron', 
                     round(np.mean(ppn_scores)*100, 2),
                     round(np.std(ppn_scores)*100, 2)]])   

algorithmdf.columns = ['Algorithm name', 'Train score mean (%)', 
                       'Train score sd (%)']
                       
# Construct table of results
from pandas.tools.plotting import table
fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, algorithmdf, loc='upper right', 
              colWidths=[0.21]*len(algorithmdf.columns))  
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
#plt.savefig('performance_table_Cross_Validation.png')
#plt.clf()
plt.show()                                                 


#####################################
# 8. LEARNING AND VALIDATION CURVES #
#####################################

# regularization parameter tuning
from sklearn.learning_curve import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
max_depth = [3, 4, 5, 6, 7, 8]
n_estimators = [5, 10, 15, 20, 25, 30]
train_scores, test_scores = validation_curve(estimator=RandomForestClassifier(),
                                             X=X_train_lda,
                                             y=y_train,
                                             param_name='max_depth',
                                             param_range=max_depth,
                                             cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)                                             

# validation curve
plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5,
         label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std,
                 test_mean - test_std, alpha=0.15,
                 color='green')
plt.grid()                 
plt.xscale('log')         
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylabel([0.7, 1.0])
plt.show()         


# performance for tuned algorithms
lr = LogisticRegression(penalty='l1', C=0.01, random_state=0)
lr.fit(X_train_lda, y_train)
lr_scores = cross_val_score(estimator=lr,
                            X=X_train_lda,
                            y=y_train,
                            cv=15)

svm = SVC(kernel='linear', C=0.01, random_state=0)
svm.fit(X_train_lda, y_train)                         
svm_scores = cross_val_score(estimator=svm,
                             X=X_train_lda,
                             y=y_train,
                             cv=15)
                         
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train_lda, y_train)
tree_scores = cross_val_score(estimator=tree,
                              X=X_train_lda,
                              y=y_train,
                              cv=15)
                         
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                max_depth=4,
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train_lda, y_train)   
forest_scores = cross_val_score(estimator=forest,
                                X=X_train_lda,
                                y=y_train,
                                cv=15)
                                            
ppn = SGDClassifier(penalty='elasticnet', loss='perceptron', n_iter=100,
                    learning_rate='optimal', random_state=0, alpha=0.001)
ppn.fit(X_train_lda, y_train)
ppn_scores = cross_val_score(estimator=ppn,
                             X=X_train_lda,
                             y=y_train,
                             cv=15)
                                                 
# table of cross-validated scores
algorithmdf = pd.DataFrame([
                    ['Logistic regression', 
                     round(np.mean(lr_scores)*100, 2),
                     round(np.std(lr_scores)*100, 2)],
                    ['Support vector machine', 
                     round(np.mean(svm_scores)*100, 2),
                     round(np.std(svm_scores)*100, 2)],
                    ['Decision tree', 
                     round(np.mean(tree_scores)*100, 2),
                     round(np.std(tree_scores)*100, 2)],
                    ['Random Forests', 
                     round(np.mean(forest_scores)*100, 2),
                     round(np.std(forest_scores)*100, 2)],
                    ['Perceptron', 
                     round(np.mean(ppn_scores)*100, 2),
                     round(np.std(ppn_scores)*100, 2)]])   

algorithmdf.columns = ['Algorithm name', 'Train score mean (%)', 
                       'Train score sd (%)']
                       
# Construct table of results
from pandas.tools.plotting import table
fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, algorithmdf, loc='upper right', 
              colWidths=[0.21]*len(algorithmdf.columns))  
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
plt.savefig('performance_table_parameter_tuned.png')
plt.clf()
#plt.show()                                                 



























































