#! /usr/bin/python3

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#read and process data
data = pd.read_csv('../data/pima-indians-diabetes.csv', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


#for_crossvalidation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

#print all crosss validation scores
print(scores)

#print mean score
print(scores.mean())