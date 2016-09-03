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

### 7. SibSp

### 8. Parch

### 9. Ticket   

### 10. Fare

### 11. cabin

Not only would the cabin number most likely have a trivial effect, it also has a very high number of missing values. 77% for the training set and 78% of the test set. These columns will be deleted during the preprocessing step.

### 12. Embarked


I am extremely suspicious of the same proportion of missing values in the test and training data sets. How did that happen? This could lead to some confounding. More on this later.




