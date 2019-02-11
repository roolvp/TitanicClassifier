from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import category_encoders as ce
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time



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

print(cross_val_score(clf, X_train, y_train, cv=7))

#TODO: Explain what all the hyperparameters do


#TODO: Use hyperparamenter tuning
from hyperopt import hp
from hyperopt import STATUS_OK

initial_params = clf.get_params()
print(initial_params)

clf.set_params(**initial_params)

#print(type(str(initial_params)))
print(cross_val_score(clf, X_train, y_train, cv=7))


def the_objective(params, X_train_s = X_train,y_train_s = y_train,X_test = X_test, y_test = y_test):
       '''
       :param params: Random Forest Params
       :param cv: Cross Validation folds
       :return:
       '''

       clf_ =  RandomForestClassifier(n_jobs = 8 )
       clf_.set_params(**params)
       print("Trying the set of paras {}".format(params))
       clf_.fit(X_train_s,y_train_s)
       y_pred = clf_.predict(X_test)
       accuracy_score =  metrics.accuracy_score(y_test, y_pred)
       print("The accuracy score is {}".format(accuracy_score))

       accuracy_inverse = 1 - accuracy_score
       #best = cv_results.mean()
       #print(cv_results)
       return {'loss': accuracy_inverse, 'params': params, 'status': STATUS_OK, 'accuracy': accuracy_score}

#define space
'''
Using this set of hyperparams

n_estimators = number of trees in the foreset
max_features = max number of features considered for splitting a node
max_depth = max number of levels in each decision tree
min_samples_split = min number of data points placed in a node before the node is split
min_samples_leaf = min number of data points allowed in a leaf node
bootstrap = method for sampling data points (with or without replacement)

Taking the Bayesian Strategy to find the best combination
'''

import numpy as np

grid_space = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


bayes_space = {'bootstrap': hp.choice('bootstrap',[True, False]),
 'max_depth': hp.choice('max_depth', np.arange(3,30, dtype= int)),
 'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(3,60, dtype= int)),
 'min_samples_split': hp.choice('min_samples_split', np.arange(3,100, dtype= int)),
 'n_estimators': hp.choice('n_estimators', np.arange(200,2000, dtype= int))
               }


#testing space
from hyperopt.pyll.stochastic import sample
print('Testing Space')
print(sample(bayes_space))

#run tunning

from hyperopt import Trials, tpe, fmin


bayes_trials = Trials()

best = fmin(fn = the_objective, space = bayes_space, algo = tpe.suggest, max_evals = 100, trials= bayes_trials)

print(best)

#print
#unnesting the parameters to columns
#Getting all results
df_best = pd.DataFrame(bayes_trials.results)

#unnesting the params dictionary
df_best = pd.concat([df_best, df_best.params.apply(pd.Series)], axis = 1, sort = False)
#dropping the dictionary column
df_best = df_best.drop(['params'], axis = 1)
print(df_best.head())

df_best['iteration'] = df_best.index + 1
ls
print(df_best.columns)

import matplotlib.pyplot as plt
import seaborn as sns

print_list = ['accuracy', 'loss', 'max_depth',
       'min_samples_leaf', 'min_samples_split', 'n_estimators', 'iteration']

fig, ax = plt.subplots()

for i,item in enumerate(print_list):
       sns.regplot('iteration', item, data = df_best)
       plt.show()

#train model with best hyperparameters

