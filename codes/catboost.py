#! /usr/bin/python3

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../data/iris.data', header=None)
#print(data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
#print(encoded_y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 7)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:\t{accuracy}')