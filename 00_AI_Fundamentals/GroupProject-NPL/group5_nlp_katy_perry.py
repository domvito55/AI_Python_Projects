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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#### 1. #########################
"""Loading the Data"""


path = "./"

filename = 'Youtube02-KatyPerry.csv'

# path = "D:/pCloudFolder/Repositories/AI_Python_Projects/COMP237_AI/GroupProject"
# filename = 'Youtube02-KatyPerry.csv'
fullpath = os.path.join(path, filename)
data_KateParry = pd.read_csv(fullpath)
pd.set_option('display.max_columns', 5)


#### 2. ########################
"""Data Exploration"""
print('#### 2. Data Exploration ########################')
print("First 3 records in the dataset:")
print(data_KateParry.head(3))
#
print('The dimension of the dataset:')
print(data_KateParry.shape)
#
print('Column names of the dataset:')
print(data_KateParry.columns.values)
#
print('Data types of each column:')
print(data_KateParry.dtypes)
#
print('The total number of cells that are null for each column:')
print(data_KateParry.isnull().sum(), "\n")  # none of the cells are null
#
data_KateParry = data_KateParry[['CONTENT', 'CLASS']]


#### 6. ########################
"""Shuffle the dataset"""
data_KateParry = data_KateParry.sample(frac=1, random_state=1)


# #### 7. ########################
"""Pandas split for the trainging data and testing data"""
train_data = data_KateParry.sample(frac=0.75, random_state=1)
test_data = data_KateParry.drop(train_data.index)


#### 3. ########################
"""prepare the data"""
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(train_data['CONTENT'])
a = count_vectorizer.get_feature_names_out()


test_tc = count_vectorizer.transform(test_data['CONTENT'])


#### 4. ########################
"""Present highlights of the output"""
print('#### 4. Present highlights of the output########################')
# shape is 262 rows * 1367 cols, here  1367 represents the total number of unique words in all the comments
print("Dimensions of training data:", train_tc.shape)

# Debug
# c = train_tc.toarray()
# print("Test:", train_tc.toarray())

# 3892 is the total number of times a word is found exist in the comment
numOfTimesWordExist = train_tc.data.size
print("Number of Times a Word Exist in training data: ", numOfTimesWordExist)

# 4438 is the total number of times a word has appeared in all the comments (if a word has appeared 10 times, it'd be recorded as 10)
frequenciesSum = train_tc.data.sum()
print("Sum of frequencies of every word in training data:", frequenciesSum, "\n")


#### 5. ########################
"""tf-idf downscale and present highlights"""

print('#### 5. Tf-idf downscale and present highlights ########################')
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
test_tfidf = tfidf.fit_transform(test_tc)
print("Dimensions of training data after idf transformation:", train_tfidf.shape)
numOfTimesWordExist_tfidf = train_tfidf.data.size
print("Number of Times a Word Exist in training data after idf transformation: ",
      numOfTimesWordExist_tfidf)
frequenciesSum_tfidf = train_tfidf.data.sum()
print("Sum of frequencies after idf transformation:", frequenciesSum_tfidf, "\n")


#### 8.########################
"""Fit the data into Naive Bayes Classifier"""
classifier = MultinomialNB().fit(train_tfidf, train_data['CLASS'])


#### 9. #######################
"""Cross validation using 5- fold and present results"""
print('#### 9. Cross validation using 5- fold and present results ########################')
# Scoring functions

num_folds = 5
accuracy_values = cross_val_score(
    classifier, train_tfidf, train_data['CLASS'], scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(
    classifier, train_tfidf, train_data['CLASS'], scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(
    classifier, train_tfidf, train_data['CLASS'], scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(
    classifier, train_tfidf, train_data['CLASS'], scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%", "\n")
# F1 score represents the balance between precision and recall

#### 10. ######################
"""Test the model using test data"""
print('#### 10. Test the model using test data ########################')
y_pred = classifier.predict(test_tfidf)
y_actual = test_data['CLASS']
confusionMatrix = confusion_matrix(y_actual, y_pred)
confusionMatrixDF = pd.DataFrame(confusionMatrix,
                                 columns=['Positive Actual', 'Negative actual'],
                                 index=['Positive pred', 'Negative pred'])
print('Test Confusion Matrix')
print(confusionMatrixDF)
print('#')
accurancy = accuracy_score(y_actual, y_pred)
print("Accuracy: " + str(round(100*accurancy, 2)) + "%", "\n")
# visualizing the confusion metrix
cmGraph = ConfusionMatrixDisplay(confusionMatrix, display_labels=None)
cmGraph.plot()
plt.title("Confusion Matrix for the test data")
plt.show()


# #### 11. ######################
"""Production - using the model to classify new commets"""
print('#### 11. Production - using the model to classify new commets ########################')
input_data = [
    "I hate this song. Katy Perry is just a waste of time",
    "I love this song. I cannot wait for the next Katy Perry clip.",
    "TBH we need more songs like this, most music nowadays are all either talking about relationships or drugs",
    "This song brings me all kinds of fierce fellings!",
    "This song sucks, I prefer Eye of The Tiger. That is a classic.",
    "Love this music video, we need more music like this in 2022. Also guys if you want to be rich like Katy Perry, check out our Funded Trading Discord here: https://traderoomplus.com/s/ytdc and we will all become millionaires!",
    "Katy Perry is great, but have you seen my new song? https://www.youtube.com/channel/UCpIHs4_NHF6EQcqVleKIjMA",
    "Join the greatest Katy Perry fun club: https://www.katyperry-fanclub.com/",
    "Are you looking for something in our Facial Cleansers store? If so, you might be interested in these items, visit:https://www.amazon.ca/?&tag=h020d3-20&ref=pd_sl_2gqjnc4wew_e&adgrpid=63852952769&hvpone=&hvptwo=&hvadid=310033452919&hvpos=&hvnetw=g&hvrand=5271324349477770641&hvqmt=e&hvdev=m&hvdvcmdl=&hvlocint=&hvlocphy=9000895&hvtargid=kwd-360364904397&hydadcr=16682_10245354&gclid=Cj0KCQiAm5ycBhCXARIsAPldzoW1VhfHr_oqKEyegbCNM6oLOhKi-hyxia_1McrDtbTCjAhoz4ZW5P4aAnfFEALw_wcB",
]

input_tc = count_vectorizer.transform(input_data)
input_tfidf = tfidf.fit_transform(input_tc)
input_pred = classifier.predict(input_tfidf)
# labeling the data as spam and non-spam, 0 = non-spam, 1 = spam
input_actual = [0, 0, 0, 0, 0, 1, 1, 1, 1]

confusionMatrix_production = confusion_matrix(input_actual, input_pred)

confusionMatrix_productionDF = pd.DataFrame(confusionMatrix_production,
                                            columns=['Positive Actual',
                                                     'Negative actual'],
                                            index=['Positive pred', 'Negative pred'])
print('Production Confusion Matrix')
print(confusionMatrix_productionDF)
# visualizing the confusion metrix
cmGraph = ConfusionMatrixDisplay(
    confusionMatrix_production, display_labels=None)
cmGraph.plot()
plt.title("Confusion Matrix for the production data")
plt.show()

print("#")
accuracy = accuracy_score(input_actual, input_pred)
print("Accuracy: " + str(round(100*accuracy, 2)) + "%", "\n")
