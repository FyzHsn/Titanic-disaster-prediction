Titanic-disaster-prediction
===========================

In this script, we predict the survivability in the Titanic disaster based on features such as gender, ticket class, family status etc. This is part of a kaggle competition.

1. We load the test and training data sets from the kaggle website.  
2. Preprocessing of the data:  
    * Remove or impute NA values.  
    * Replace ordinal and nominal features with numerical values.  
3. Exploratory data analysis: What does the data look like? Form intuition!   
4. Train a machine learning algorithm.     
5. Test the algorithm.  
6. Try other algorithms and ways to improve them.  

Exploratory data analysis
-------------------------

We are given a training data set and a test data set.
Test data set dimension - (418, 11)  
Train data set dimension - (891, 12)  
We have almost a 70-30 training to test data split. 68-32 split to be more precise.   
Survival takes on binary values of 0 (die) or 1 (live) and shown as a function of the following features.

### 1. passengerid

This is an enumeration of the passenger. However, it is not useful for prediction of survivability and will be dropped after preprocessing.

### 2. survived

This column takes values of 0 (dead) and 1 (survived) values. There are no missing values in either the training and test sets.

### 3. pclass

This is an ordinal value that indicates the class of the passenger accomodation. I imagine, there would be a non-trivial correlation, with survival and this variable.

### 4. name

This is a categorical label that should be ignored. This column will be dropped since this cannot have a generalizable pattern. 

### 5. sex

Takes on values: male and female. No missing values for this category.

### 6. age  

I imagine this is an important variable, since children would have a higher chance of being on the boats due to societal norms. However, this column has a lot of missing values for both the training (20%) and the test sets (20%).  

### 7. sibsp

This variable tells you the number of siblings or spouses aboard. It has no missing values.

### 8. parch

This variable tells you the number of parents and children on the ship. There are no missing values.

### 9. ticket   

Ticket number. This is a non-consequential variable and will be deleted in the preprocessing step. 

### 10. fare

The fare is a function of two variables that are taken account of: Point of the journey on which they got on (there were 3 steps) and cabin class. Since, this is already taken care of, I will not be using this column as well. 

### 11. cabin

Not only would the cabin number most likely have a trivial effect, it also has a very high number of missing values. 77% for the training set and 78% of the test set. These columns will be deleted during the preprocessing step.

### 12. embarked

This variable tells us the point of embarkation: C = Cherbourg; Q = Queenstown; S = Southampton. I will also be deleting this variable as well.

I am extremely suspicious of the same proportion of missing values in the test and training data sets. How did that happen? This could lead to some confounding. More on this later.

### Survival rate for different groups

In the training set, 19% of men survived while 74% of women survived.  
In the test set, 0% of mean survived while 100% of women survived. This makes me severely question how the data was split into training and test data sets. To be honest, because of this, I do not believe that doing really well with the test data set is even possible.  

In the training set, 63% of class 1, 47% of class 2 and 24% of class 3 people survived.
In the test set, 47% of class 1, 32% of class 2 and 33% of class 3 people survived. 

Using the random forest approach we find the importance of features.  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_random_forest.png?raw=true)  

The l1 norm regularization approach can also be used to look at the importance of features as well by checking the sparseness of the weight coefficients.  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_l1norm.png?raw=true)  

The sequential backward selection algorithm is a third approach to finding out the important features to select. Using intuitive ideas, however, I have already done the feature selection.  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_SBS_svm.png?raw=true)  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_SBS_knn.png?raw=true)  

In the table below, we show the performance of each model on the test sets. Of course, this is a very rudimentary arrangement. For a careful appraisal of the training models, the training data set needs to be broken down further into a validation set and the training set. Then, based on the performance on the validation set, we can pick the superior model and apply it to the test dataset. We will utilize these subtlers method later.

![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/performance_table.png?raw=true)

Principal component analysis (PCA)
----------------------------------

Unsupervised data compression.
Another non-linear variation is the kernel proncipal component analysis.
PCA doesn't improve the performance. The features that I have kept already are optimal.  

Linear discriminant analysis (LDA)
----------------------------------

Supervised dimensionality reduction technique for maximum class separability. This leads to better performance of the codes and reduces overfitting.  

![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/performance_table_LDA.png?raw=true)  

With a few exceptions, LDA usually performs better than PCA. If a class contains a small number of samples, PCA according to A. M. Martinez performs better in the context of image recognition. 

Model evaluation and hyperparameter tuning
==========================================

Cross-validation to assess performance
--------------------------------------

Here is a fantastic case for using cross-validation methods. Whereas previously Random Forests had the highest training score combined with the lowest test score, using cross-validation, we find that Random Forests perform the worst. 

Support vector machines or Logistic regression are the best performers according to the cross-validation scores.  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/performance_table_Cross_Validation.png?raw=true)    

Learning and validation curves
------------------------------

As can be seen in the following table, I was able to improve the performance of one of the algorithms using validation curves to tune the parameters. Playing with the maximum depth of Random Forest Classifier, I was able to improve its classification accuracy while avoiding overfitting. At the present moment, this algorithm with the parameters is the best classifier. 

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                max_depth=4,
                                random_state=1,
                                n_jobs=-1)

![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/performance_table_parameter_tuned.png?raw=true)  

Tuning ML algorithms via grid search
------------------------------------


Ensemble learning
=================

Majority voting
---------------

Bagging
-------

Adaptive boosting
-----------------


Data Analysis Steps
===================

1. Clean and pre-process data sets.   
2. Feature selection + transformation.    
      * Study the importance of each feature.   
      * Study the effect of LDA.    
      * Using a combination of feature selection and transformation, process the features of the data-set.   
3. Use cross-validation to find the performance of various algorithms on the dataset in the classification task:    
      * Perceptron, adaline, logistic regression, support vector machine, decision tree learning.  
      * Ensemble learning via majority voting, bagging or boosting.   
4. Tune hyper-parameters via grid search.    
5. Voila, you have your model. After running through test sets, feed it into the training algorithm.  


Remember that use to test set to only conclude if the data is being overfit or underfit. Use cross-validation to choose optimally performing algorithm.   





