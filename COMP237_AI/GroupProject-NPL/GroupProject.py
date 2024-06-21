# -*- coding: utf-8 -*-

"""

Created on Tue Nov 29 13:51:05 2022



@authors:Song Malisa Se(301233051), Matheus Teixeira (301236904), Viet Hoang, Yi-lin Lou (301226659), Yin-Siang Mao (301180968)

Comp 237 - Sec. 002

Group 5

Group Assignment - NPL

"""

# Load modules

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

#### 1. #########################

path = "D:/pCloudFolder/Repositories/Semester3/AI_Python_Projects/COMP237_AI/GroupProject"
filename = 'Youtube02-KatyPerry.csv'
fullpath = os.path.join(path,filename)
data_KateParry = pd.read_csv(fullpath)
pd.set_option('display.max_columns', 5)


#### 2. ########################
print('#### 2. ########################')
print(data_KateParry.head(3))
#
print('#')
print(data_KateParry.shape)
#
print('#')
print(data_KateParry.columns.values)
#
print('#')
print(data_KateParry.dtypes)
#
print('#')
print(data_KateParry.isnull().sum(),"\n")
#
data_KateParry = data_KateParry[['CONTENT','CLASS']]


#### 6. ########################
data_KateParry = data_KateParry.sample(frac=1, random_state=1)


# #### 7. ########################
train_data = data_KateParry.sample(frac=0.75, random_state=1)
test_data = data_KateParry.drop(train_data.index)


#### 3. ########################
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(train_data['CONTENT'])
test_tc = count_vectorizer.transform(test_data['CONTENT'])


#### 4. ########################
print('#### 4. ########################')
#shape is 350 rows * 1738 cols, here 1738 represents the total number of unique words in all the comments the a unique word is found in a comment                 regardless whether the word has been repeated in the same comment
print("Dimensions of training data:", train_tc.shape) 

# Debug
# c = train_tc.toarray()
# print("Test:", train_tc.toarray())

# 5275 is the total number of times
numOfTimesWordExist = train_tc.data.size    
print("Number of Times a Word Exist in training data: ", numOfTimesWordExist)

# 6177 is the total number of times a word has appeared in all the comments (if a word has appeared 10 times, it'd be recorded as 10)
frequenciesSum = train_tc.data.sum() 
print("Sum of frequencies of every word in training data:", frequenciesSum,"\n")


#### 5. ########################
print('#### 5. ########################')
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
test_tfidf = tfidf.fit_transform(test_tc)
print("Dimensions of training data after idf transformation:", train_tfidf.shape)
numOfTimesWordExist_tfidf = train_tfidf.data.size
print("Number of Times a Word Exist in training data after idf transformation: ", numOfTimesWordExist_tfidf)
frequenciesSum_tfidf = train_tfidf.data.sum()
print("Sum of frequencies after idf transformation:", frequenciesSum_tfidf,"\n")
#########explain how 6177 turn into 1177.5939...


#### 8.########################
classifier = MultinomialNB().fit(train_tfidf, train_data['CLASS'])


#### 9. #######################
print('#### 9. ########################')
# Scoring functions

num_folds = 5
accuracy_values = cross_val_score(classifier, train_tfidf, train_data['CLASS'], scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier, train_tfidf, train_data['CLASS'], scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier, train_tfidf, train_data['CLASS'], scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier, train_tfidf, train_data['CLASS'], scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%","\n")


#### 10. ######################
print('#### 10. ########################')
y_pred   = classifier.predict(test_tfidf)
y_actual = test_data['CLASS']
confusionMatrix = confusion_matrix(y_actual, y_pred)
confusionMatrix = pd.DataFrame(confusionMatrix,
                  columns = ['Positive Actual','Negative actual'],
                  index=['Positive pred','Negative pred'])
print('Test Confusion Matrix')
print (confusionMatrix)
print('#')
accurancy = accuracy_score(y_actual, y_pred)
print("Accuracy: " + str(round(100*accurancy, 2)) + "%","\n")


# #### 11. ######################
print('#### 11. ########################')
input_data = [
  "I hate this song. Katy Perry is just a waste of time",
  "I love this song. I cannot wait for the next Katy Perry clip.",
  "This song brings me all kinds of fierce fellings!",
  "This song sucks, I prefer Eye of The Tiger. That is a classic.",
  "Katy Perry is great, but have you seen my new song? https://www.youtube.com/channel/UCpIHs4_NHF6EQcqVleKIjMA",
  "Join the greatest Katy Perry fun club: https://www.katyperry-fanclub.com/"
  ]

input_tc = count_vectorizer.transform(input_data)
input_tfidf = tfidf.fit_transform(input_tc)
input_pred   = classifier.predict(input_tfidf)
input_actual = [0,0,0,0,1,1]

confusionMatrix_production = confusion_matrix(input_actual, input_pred)

confusionMatrix_production = pd.DataFrame(confusionMatrix_production,
                  columns = ['Positive Actual','Negative actual'],
                  index=['Positive pred','Negative pred'])
print('Production Confusion Matrix')
print (confusionMatrix_production)
print("#")
accurancy = accuracy_score(input_actual, input_pred)
print("Accuracy: " + str(round(100*accurancy, 2)) + "%","\n")
