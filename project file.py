# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:07:20 2022

@author: chand
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer = pd.read_excel(r"A:\CHAND\Downloads\sampla_data_08_05_2022(final).xlsx")
sample = pd.read_excel(r"A:\CHAND\Downloads\sampla_data_08_05_2022(final).xlsx")
cancer.columns
cancer.info()

#*******************************************EDA**************************************************
cancer.describe()
cancer.skew()
cancer.kurt()


#Duplicates
cancer.duplicated().sum()

#missing values
cancer.isna().sum()

#zero variance
cancer.var()

cancer = cancer.drop("Patient_ID", axis = 1)  # drop the nominal features
cancer = cancer.drop("Mode_Of_Transport", axis = 1)  # drop the nominal features
cancer = cancer.drop("Sample_Collection_Date", axis = 1)  # drop the nominal features
cancer = cancer.drop("Test_Booking_Date", axis = 1)  # drop the nominal features


#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
#X = df.iloc[:, 0:9]


cols = [ 'Patient_Gender', 'Test_Name', 'Sample',
       'Way_Of_Storage_Of_Sample',    
       'Cut-off Schedule',
         'Traffic_Conditions', 'Reached_On_Time']

#
# Encode labels of multiple columns at once
#
cancer[cols] = cancer[cols].apply(LabelEncoder().fit_transform)
#
# Print head
#
cancer.head()






#Normalization
def normfunc(val):
    a = (val - val.min())/(val.max() - val.min())
    return(a)

cancer_norm = normfunc(cancer.loc[:,cancer.columns != 'Reached_On_Time'])
y = cancer.Reached_On_Time
y.value_counts()



# let's find outliers in Salaries
sns.boxplot(cancer.Patient_Age) # OUTliers

sns.boxplot(cancer.Test_Booking_Time_HH_MM )

sns.boxplot(cancer['Cut-off time_HH_MM']) #  outliers

sns.boxplot(cancer.Scheduled_Sample_Collection_Time_HH_MM)#no outliers

sns.boxplot(cancer.Agent_Location_KM )  #outliers

sns.boxplot(cancer.Time_Taken_To_Reach_Patient_MM )#  outliers

sns.boxplot(cancer.Time_For_Sample_Collection_MM) #outliers

sns.boxplot(cancer.Lab_Location_KM) #outliers

sns.boxplot(cancer.Time_Taken_To_Reach_Lab_MM )#outliers


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

df= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Test_Booking_Time_HH_MM'])

cancer= winsor.fit_transform(cancer[['Test_Booking_Time_HH_MM']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Test_Booking_Time_HH_MM)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer.Patient_Age= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='right', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

cancer= winsor.fit_transform(cancer[['Patient_Age']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(cancer.Patient_Age)


















#**************************************Ensemble model***************************************

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, Y_train, Y_test = train_test_split(cancer_norm, y,stratify = y, test_size = 0.2)

#*****************BAGGING******************************
from sklearn.ensemble import BaggingClassifier

model_dt = DecisionTreeClassifier()   #Initialising decision tree
bag = BaggingClassifier(base_estimator = model_dt, n_estimators = 5000,
                            bootstrap = True, n_jobs = 1, random_state = 42) #initialising Bagging 

bag.fit(X_train, Y_train)

#Prediction on test data
test_pred = bag.predict(X_test)

#Calculating accuracy of train and test
test_acc = accuracy_score(Y_test, test_pred)

train_pred = bag.predict(X_train)   #prediction on train data
train_acc = accuracy_score(Y_train, train_pred)

print("Train accuracy:",train_acc)
print("Test accuracy:",test_acc)

#Confusion matrix
pd.crosstab(train_pred, Y_train, rownames = ['Actual'], colnames= ['Predictions'])
pd.crosstab(test_pred, Y_test, rownames = ['Actual'], colnames= ['Predictions'])


#********************Boosting*****************************
from xgboost import XGBClassifier

model_xgb = XGBClassifier(max_depths = 8, n_estimators = 100000, learning_rate = 0.1, n_jobs = -1) 

model_xgb.fit(X_train, Y_train)

#Prediction on test data
test_pred = model_xgb.predict(X_test)

#Calculating accuracy of train and test
test_acc = accuracy_score(Y_test, test_pred)

train_pred = model_xgb.predict(X_train)   #prediction on train data
train_acc = accuracy_score(Y_train, train_pred)

print("Train accuracy:",train_acc)
print("Test accuracy:",test_acc)

#Confusion matrix
pd.crosstab(train_pred,Y_train.iloc[:,0], rownames = ['Actual'], colnames= ['Predictions'])
pd.crosstab(test_pred, Y_test.iloc[:,0], rownames = ['Actual'], colnames= ['Predictions'])


#***********************Voting**********************************
# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier


# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

#****Hard Voting*****
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(X_train, Y_train)

# Predict the most voted class
hard_predictions = voting.predict(X_test)

# Accuracy of hard voting
print('Hard Voting:(Test)', accuracy_score(Y_test, hard_predictions))
print('Hard Voting:(Train)', accuracy_score(Y_train, voting.predict(X_train)))


# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(X_train, Y_train)
learner_4.fit(X_train, Y_train)
learner_5.fit(X_train, Y_train)
learner_6.fit(X_train, Y_train)

# Predict the most probable class
soft_predictions1 = voting.predict(X_test)
soft_predictions2 = voting.predict(X_train)
# Get the base learner predictions
predictions_4 = learner_4.predict(X_test)
predictions_5 = learner_5.predict(X_test)
predictions_6 = learner_6.predict(X_test)

# Accuracies of base learners
print('L4:', accuracy_score(Y_test, predictions_4))
print('L5:', accuracy_score(Y_test, predictions_5))
print('L6:', accuracy_score(Y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting(Test):', accuracy_score(Y_test, soft_predictions1))
print('Soft Voting(Train):', accuracy_score(Y_train, soft_predictions2))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(cancer_norm, y, test_size = 0.2)

model_dt = DT()
model_dt.fit(X_train,Y_train)

#Prediction on test input
test_pred = model_dt.predict(X_test)

#Calculating test and train accuracy
test_acc = accuracy_score(test_pred,Y_test)

train_pred = model_dt.predict(X_train)  #prediction on X train to find train accuracy
train_acc = accuracy_score(train_pred, Y_train)

print("Train accuracy:",train_acc)
print("Test accuracy:",test_acc)

pd.crosstab(train_pred, Y_train, rownames = ['Actual'], colnames= ['Predictions'])
pd.crosstab(test_pred, Y_test, rownames = ['Actual'], colnames= ['Predictions'])

#Train accuracy is high and test accuracy is low  leads to overfitting model
#So Random forest model will be done to handle overfit

#*******************************Random Forest****************************************************
from sklearn.ensemble import RandomForestClassifier

cancer_rf = RandomForestClassifier(n_estimators=50000, n_jobs=5, random_state=42, max_depth = 3)
cancer_rf.fit(X_train,Y_train)

#Prediction on test input
test_pred = cancer_rf.predict(X_test)

#Calculating test and train accuracy
test_acc = accuracy_score(test_pred,Y_test)

train_pred = cancer_rf.predict(X_train)  #prediction on X train to find train accuracy
train_acc = accuracy_score(train_pred, Y_train)

print("Train accuracy:",train_acc)
print("Test accuracy:",test_acc)

pd.crosstab(train_pred, Y_train, rownames = ['Actual'], colnames= ['Predictions'])
pd.crosstab(test_pred, Y_test, rownames = ['Actual'], colnames= ['Predictions'])

#********************************************************************************************8
from sklearn.model_selection import KFold

X = cancer_norm   #input features

k = 5
kf = KFold(n_splits=k, random_state=None)
model = RandomForestClassifier(n_estimators=5000, n_jobs=3, random_state=42)
acc_score = []   # to store test accurace scores
Tacc_score = []  # to store train accuarcy scores

for train_index , test_index in kf.split(X):
   
    #Split the dataset into train and test
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

    #Train the model
    model.fit(X_train,y_train)
    
    #Predict using the model
    pred_values = model.predict(X_test)
    pred_train = model.predict(X_train)

    #Evaluate the model
    test_acc = accuracy_score(pred_values , y_test)
    train_acc = accuracy_score(pred_train, y_train)
    
    print("\nTest Accuracy: ", test_acc)
    acc_score.append(test_acc)
    print("\nTrain Accuracy: ", train_acc)
    Tacc_score.append(train_acc)
    
print("Train accuracy:", np.mean(Tacc_score))
print("Test accuracy:", np.mean(acc_score))


# saving the model
# importing pickle
import pickle
pickle.dump(model_dt, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(cancer.iloc[0:16,:16])
list_value

print(model.predict(list_value))
