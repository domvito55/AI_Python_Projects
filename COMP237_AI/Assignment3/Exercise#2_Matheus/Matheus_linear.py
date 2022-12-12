# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:20:53 2022

@author: Matheus Teixeira
301236904
Comp 237 - Sec. 002
Week 5 - Exercise 2
"""
# Load modules
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

#### a. #########################
path = "D:/Dropbox/Canada/College/0_Cursos/3_Semestre/COMP237_AI/Assignments/Assignment3/Exercise#2_Matheus"
filename = 'Ecom Expense.csv'
fullpath = os.path.join(path,filename)
ecom_exp_Matheus = pd.read_csv(fullpath)
pd.set_option('display.max_columns', 5)

#### b.i ########################
print('#### b.i ########################')
print(ecom_exp_Matheus.head(3))
#### b.ii #######################
print('#### b.ii #######################')
print(ecom_exp_Matheus.shape)
#### b.iii ######################
print('#### b.iii ######################')
print(ecom_exp_Matheus.columns.values)
#### b.iv #######################
print('#### b.iv #######################')
print(ecom_exp_Matheus.dtypes)
#### b.v ########################
print('#### b.v ########################')
print(ecom_exp_Matheus.isnull().sum())

print('#### c. #########################')
#### c.i ########################
cat_var = []
dummy_names = []
for name, dtype in ecom_exp_Matheus.dtypes.items():
  if dtype.type is np.object_:
    cat_var.append(name)
cat_var.remove('Transaction ID')
for var in cat_var:
    cat_list = pd.get_dummies(ecom_exp_Matheus[var], prefix=var)
    dummy_names += cat_list.columns.values.tolist()
    #### c.ii #######################
    ecom_exp_Matheus = ecom_exp_Matheus.join(cat_list)
    #### c.ii & c.iii ###############
    ecom_exp_Matheus = ecom_exp_Matheus.drop(var, axis=1)
    
#### c.iv #######################
ecom_exp_Matheus = ecom_exp_Matheus.drop('Transaction ID', axis=1)

#Debug
#print(ecom_exp_Matheus.head(5))
#print(ecom_exp_Matheus.shape)

#### c.v #######################
def normalize(dataframe):
  #Debug
  # print(dataframe.max())
  # print(dataframe.min())
  # print(dataframe.max()-dataframe.min())
  # print("\n -------------------- \n")
  # print(dataframe)
  # print((dataframe-dataframe.min())/(dataframe.max()-dataframe.min()))
  return (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())

#### c.vi ######################
ecom_exp_Matheus = normalize(ecom_exp_Matheus)

#### c.vii #####################
print('#### c.vii ######################')
print(ecom_exp_Matheus.head(2))

#### c.viii ####################
ecom_exp_Matheus.hist(figsize=(9,10))

#### c.ix ######################
scatter_vars = ['Age','Monthly Income','Transaction Time','Total Spend']
pd.plotting.scatter_matrix(ecom_exp_Matheus[scatter_vars], alpha=0.4, figsize=(13,15))

#### d.i ######################
toKeep = ['Monthly Income', 'Transaction Time'] + dummy_names
X = ecom_exp_Matheus[toKeep]
Y = ecom_exp_Matheus['Total Spend']

#### d.ii #####################
#### d.iii ####################
#### d.iv #####################
x_train_Matheus, x_test_Matheus, y_train_Matheus, y_test_Matheus = train_test_split(X, Y, test_size=0.35, random_state=4)

#### d.v ######################
lm = linear_model.LinearRegression()
lm.fit(x_train_Matheus, y_train_Matheus)

#### d.vi ######################
print('#### d.vi #######################')
print (f"Weights: {lm.coef_}")

#### d.vii #####################
print('#### d.vii ######################')
print(f"R^2:  {lm.score(x_train_Matheus, y_train_Matheus)}")



#### d.viii ####################
#### d.i ######################
toKeep = ['Monthly Income', 'Transaction Time', 'Record'] + dummy_names
X = ecom_exp_Matheus[toKeep]
Y = ecom_exp_Matheus['Total Spend']

#### d.ii #####################
#### d.iii ####################
#### d.iv #####################
x_train_Matheus, x_test_Matheus, y_train_Matheus, y_test_Matheus = train_test_split(X, Y, test_size=0.35, random_state=4)

#### d.v ######################
lm = linear_model.LinearRegression()
lm.fit(x_train_Matheus, y_train_Matheus)

#### d.vi ######################
print('#### d.viii #####################')
print (f"Weights: {lm.coef_}")

#### d.vii #####################
print(f"R^2:  {lm.score(x_train_Matheus, y_train_Matheus)}")


