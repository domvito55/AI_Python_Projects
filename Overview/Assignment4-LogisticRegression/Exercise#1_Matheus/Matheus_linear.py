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
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import metrics
#### b.4 #######################
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import matplotlib.pyplot as plt


#### a. #########################
# path = os.path.dirname(os.path.realpath(__file__))
  # the above approach only works if you run the file, it wont work if you run
  # it line by line
path = 'D:/pCloudFolder/Repositories/AI_Python_Projects/COMP237_AI/Assignment4/Exercise#1_Matheus'
filename = 'titanic.csv'
fullpath = os.path.join(path,filename)
titaninc_Matheus = pd.read_csv(fullpath)
pd.set_option('display.max_columns', 10)

#### b.1 ########################
print('#### b.1 ########################')
print(titaninc_Matheus.head(3))
#### b.2 ########################
print('#### b.2 ########################')
print(titaninc_Matheus.shape)
#### b.3 ########################
print('#### b.3 ########################')
print(titaninc_Matheus.info())
#### b.4 ########################
print('#### b.4 ########################')
print(titaninc_Matheus['Sex'].unique())
print(titaninc_Matheus['Pclass'].unique())

#### c.1.a #####################
byClassTable = pd.crosstab(titaninc_Matheus.Pclass,
                           titaninc_Matheus.Survived)
byClassGraph = byClassTable.div(byClassTable.sum(1).astype(float),
                                axis=0).plot(kind='bar', stacked=True,
                                             figsize=(10,10))
handles, labels = byClassGraph.get_legend_handles_labels()
byClassGraph.legend(handles, ['Died', 'Survived'], prop={'size': 15})
plt.suptitle('Proportion of titanic survivors by passenger class',
             weight='bold', size=20)
plt.title('A graphic by Matheus', {'fontsize': 20})
plt.xlabel('Passenger Class', {'fontsize': 20})
plt.ylabel('Proportion of people who survived/died', {'fontsize': 20})
plt.xticks(rotation = 0)
byClassGraph.tick_params(axis='both', which='major', labelsize=15)

#### c.1.b #####################
byGenderTable = pd.crosstab(titaninc_Matheus.Sex,
                            titaninc_Matheus.Survived)
byGenderGraph = byGenderTable.div(byGenderTable.sum(1).astype(float),
                                  axis=0).plot(kind='bar', stacked=True,
                                               figsize=(10,10))
handles, labels = byGenderGraph.get_legend_handles_labels()
byGenderGraph.legend(handles, ['Died', 'Survived'], prop={'size': 15})
plt.suptitle('Proportion of titanic survivors by gender',
             weight='bold', size=20)
plt.title('A graphic by Matheus', {'fontsize': 20})
plt.xlabel('Gender', {'fontsize': 20})
plt.ylabel('Proportion of people who survived/died', {'fontsize': 20})
plt.xticks(rotation = 0)
byGenderGraph.tick_params(axis='both', which='major', labelsize=15)

#### c.2 #######################
scatter_vars = ['Survived','Sex','Pclass','Fare', 'SibSp', 'Parch']
pd.plotting.scatter_matrix(titaninc_Matheus[scatter_vars], alpha=0.4, figsize=(13,15))

#### d.1 #######################
titaninc_Matheus = titaninc_Matheus.drop('PassengerId', axis=1)
titaninc_Matheus = titaninc_Matheus.drop('Name', axis=1)
titaninc_Matheus = titaninc_Matheus.drop('Ticket', axis=1)
titaninc_Matheus = titaninc_Matheus.drop('Cabin', axis=1)

#### d.2 #######################
cat_var = ['Sex', 'Embarked']
dummy_names = []
for var in cat_var:
    cat_list = pd.get_dummies(titaninc_Matheus[var], prefix=var)
    dummy_names += cat_list.columns.values.tolist()
    #### d.3 ########################
    titaninc_Matheus = titaninc_Matheus.join(cat_list)
    #### d.3 & d.4 ##################
    titaninc_Matheus = titaninc_Matheus.drop(var, axis=1)

#### d.5 #######################
NuNserie = titaninc_Matheus['Age'].isnull()
index = 0
for NuN in NuNserie:
    if(NuN):
      titaninc_Matheus['Age'][index] = titaninc_Matheus['Age'].mean()
    index += 1
  
#### d.6 #######################
titaninc_Matheus = titaninc_Matheus.astype(float)

#### d.7 #######################
print(titaninc_Matheus.info())

#### d.8 #######################
def normalize(dataframe):
  return (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())

#### d.9 #######################
titaninc_Matheus = normalize(titaninc_Matheus)

#### d.10 ######################
print('#### d.10 #######################')
print(titaninc_Matheus.head(2))

#### d.11 ######################
titaninc_Matheus.hist(figsize=(9,10))

#### d.13 ######################
data_column_names = titaninc_Matheus.columns.values.tolist()
x = data_column_names[1:]
y = data_column_names[0]
x_Matheus = titaninc_Matheus[x]
y_Matheus  = titaninc_Matheus[y]

#### d.13.i ####################
x_train_Matheus, x_test_Matheus, y_train_Matheus, y_test_Matheus = train_test_split(x_Matheus, y_Matheus, test_size=0.30, random_state=4)

#### e.1 #######################
Matheus_model = linear_model.LogisticRegression(solver='lbfgs')
Matheus_model.fit(x_train_Matheus, y_train_Matheus)

#### e.2 #######################
print('#### e.2 #######################')
nice_table = pd.DataFrame(zip(x_train_Matheus.columns, np.transpose(Matheus_model.coef_)))
nice_table.columns = ['Feature', 'Weight']
print(nice_table)

#### e.3.1 #####################
#### e.3.2 #####################
print('#### e.3.1 #####################')
print('#### e.3.2 #####################')

scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'),
                         x_train_Matheus, y_train_Matheus,
                         scoring='accuracy', cv=10)

values = pd.DataFrame([scores.min(), scores.mean(), scores.max()],
                  index = ['Min score','Mean score','Max score'],
                  columns = ['run 1 (30%)'])

print(values)

#### e.3.3 #####################
print('#### e.3.3 #####################')
for i in range(10,51,5):
  if(i == 30):
    seed = 4
  else:
    seed = i
  x_train_Matheus, x_test_Matheus, y_train_Matheus, y_test_Matheus = train_test_split(x_Matheus, y_Matheus, test_size=i/100, random_state=4)
  
  scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'),
                         x_train_Matheus, y_train_Matheus,
                         scoring='accuracy', cv=10)
  
  to_join = pd.DataFrame([scores.min(), scores.mean(), scores.max()],
                     index = ['Min score','Mean score','Max score'],
                     columns = ['run ' + str(i) + '%'])
  
  values = values.join(to_join)

print(values)

#### b.1 #######################
x_train_Matheus, x_test_Matheus, y_train_Matheus, y_test_Matheus = train_test_split(x_Matheus, y_Matheus, test_size=0.3, random_state=4)
Matheus_model.fit(x_train_Matheus, y_train_Matheus)
#### b.2 #######################
y_pred_Matheus = Matheus_model.predict_proba(x_test_Matheus)
#### b.3 #######################
y_pred_Matheus_flag = y_pred_Matheus[:,1] > 0.5
y_pred_Matheus = y_pred_Matheus_flag
#### b.5 #######################
print('#### b.5 ########################')
print ('Accurancy: ', accuracy_score(y_test_Matheus,
                                    y_pred_Matheus))
#### b.6 #######################
print('#### b.6 ########################')
print ('Confusion Matrix:\n', confusion_matrix(y_test_Matheus,
                                    y_pred_Matheus))
#### b.7 #######################
print('#### b.7 ########################')
print ('Classificatio report:\n', classification_report(y_test_Matheus, y_pred_Matheus))

#### b.9 #######################
#### b.2 #######################
y_pred_Matheus = Matheus_model.predict_proba(x_test_Matheus)
#### b.3 #######################
y_pred_Matheus_flag = y_pred_Matheus[:,1] > 0.75
y_pred_Matheus = y_pred_Matheus_flag
#### b.5 #######################
print('#### b.5 ########################')
print ('Accurancy: ', accuracy_score(y_test_Matheus,
                                    y_pred_Matheus))
#### b.6 #######################
print('#### b.6 ########################')
print ('Confusion Matrix:\n', confusion_matrix(y_test_Matheus,
                                    y_pred_Matheus))
#### b.7 #######################
print('#### b.7 ########################')
print ('Classificatio report:\n', classification_report(y_test_Matheus, y_pred_Matheus))




