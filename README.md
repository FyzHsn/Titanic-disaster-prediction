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

![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_random_forest.png?raw=true)  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_l1norm.png?raw=true)  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_SBS_svm.png?raw=true)  
![](https://github.com/FyzHsn/Titanic-disaster-prediction/blob/master/Figs/Feat_importance_SBS_knn.png?raw=true)  

![](?raw=true)

