# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:28:02 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("D:\\chetan\\assignment\\5.logitic regration\\creditcard.csv")
data.info()
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
colum = make_column_transformer((OneHotEncoder(),['card','owner','selfemp']),remainder='passthrough')
data.info()
data.head()
colum.fit_transform(data)
data.info()
data.head()

data.columns

from sklearn.preprocessing import LabelEncoder
a= data.dtypes==object
print(a)
le=LabelEncoder()
b = data.columns[a].tolist()
print(b)
data[b]=data[b].apply(lambda col: le.fit_transform(col))
data[b].head(10)
data.info()
data.describe()
data.isnull().sum()
data.head()
X=data[['reports', 'age', 'income', 'share',
       'expenditure', 'owner', 'selfemp', 'dependents', 'months', 'majorcards',
       'active']]
y= data[['card']]
X.shape
X_train = X.iloc[0:1000,:]
y_train = y.iloc[:1000,:]
X_test = X.iloc[1001:,:]
y_test = y.iloc[1001:,:]
logic = LogisticRegression()

logic.fit(X_train,y_train)
logic.score(X_train,y_train)
logic.predict(X_test)
logic.score(X_test, y_test)
