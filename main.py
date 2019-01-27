from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import category_encoders as ce
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


#TODO: Train Randon forest classifiers

#read the data
data_path = 'data/train.csv'
TrainDF = pd.read_csv(data_path)
print(TrainDF.head())

print(TrainDF.columns)
#potenfial config columns
#ignoring cabin
feature_columns = ['age', 'embarked', 'fare', 'name', 'parch', 'pclass',
       'sex', 'sibsp']
target_column = ['survived']

#split train_test
X_train, X_test, y_train, y_test = train_test_split(TrainDF[feature_columns], TrainDF[target_column].values.ravel())


print(len(X_train))

print(len(X_test))

print(len(y_train))

print(len(y_test))


#X_train = TrainDF[feature_columns]

#convert to np array
#y_train = TrainDF[target_column].values.ravel()

#labelendoding
categorical_columns = ['embarked', 'name', 'parch', 'pclass',
       'sex']
# Labels are the values we want to predict
OEncoder = ce.ordinal.OrdinalEncoder(cols=categorical_columns)
X_train = OEncoder.fit_transform(X_train)
X_train = X_train.fillna(X_train.mean())

X_test = OEncoder.transform(X_test)
X_test = X_test.fillna(X_train.mean())


#define classifier
clf = RandomForestClassifier()
clf.fit(X = X_train,y = y_train)



print(X_train.columns)
importances = clf.feature_importances_
print(importances)

y_pred = clf.predict(X_test)

print(len(y_pred))

print(len(y_test))
#accuracy
print(metrics.accuracy_score(y_test,y_pred))

print(metrics.roc_auc_score(y_test,y_pred))
#TODO: perform cross validation

print(cross_val_score(clf, X_train, y_train, cv=3))

#TODO: Explain what all the hyperparameters do


#TODO: Use hyperparamenter tuning
from hyperopt import hp
from hyperopt import STATUS_OK

initial_params = clf.get_params()
print(initial_params)

def objective(params, cv = 3, X_train, y_train):
       '''
       :param params: Random Forest Params
       :param cv: Cross Validation folds
       :return:
       '''

       clf =  RandomForestClassifier()
       clf.set_params(params)
       cross_val_score(clf, X_train, y_train, cv)


       #cv_results =


#TODO: Train Logistic Regression

#TODO: Compare Feature Performances

#TODO: Create a Pipeline at least one class

#TODO: Include bag of words




