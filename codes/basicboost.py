#! /usr/bin/python3

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#read and process data
data = pd.read_csv('../data/pima-indians-diabetes.csv', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)


#model defn 
model = XGBClassifier()
model.fit(X_train, y_train)

#make prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy:\t{accuracy}')
