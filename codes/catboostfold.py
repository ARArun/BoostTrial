#! /usr/bin/python3

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('../data/iris.data', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)

model = XGBClassifier()
scores = cross_val_score(model, X, encoded_y, cv=10, scoring='accuracy')

print(f'Mean Accuracy:\t{scores.mean()*100}%')