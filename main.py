from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd



#TODO: Train Randon forest classifiers

#read the data
data_path = 'data/train.csv'
TrainDF = pd.read_csv(data_path)
print(TrainDF.head())

print(TrainDF.columns)

#potenfial config columns
feature_columns = ['age', 'cabin', 'embarked', 'fare', 'name', 'parch', 'pclass',
       'sex', 'sibsp']

#potential config file column
target_column = ['survived']

X_train = TrainDF[feature_columns]
Y_train = TrainDF[target_column]

#define classifier
clf = RandomForestClassifier()

clf.fit(X = X_train,y = Y_train)


#perform cross validation


#get feature importances




#TODO: Use hyperparamenter tuning
#TODO: Explain what all the hyperparameters do
#TODO: Train Logistic Regression
#TODO: Compare Feature Performances
#TODO: Include bag of words


