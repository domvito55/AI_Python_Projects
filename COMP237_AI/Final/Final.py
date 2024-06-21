# -*- coding: utf-8 -*-
"""
Created on Dec 14 2022
@Author: Matheus Teixeira
Title: Final Exam
Neural network
"""

# Load modules
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords





import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

#### 1. #########################

corpus_Matheus = "The Artificial Intelligence field has played an amazing role in the world's society, the past decade. We can attribute this to the researchers, who work in field. Whom have dedicated their time and efforts to advance the field. Their contributions in the fields of NLP, computer vision, medical research and many others. I would love to have a role in it. NLP is still at the beginning and there is still a lot to be discovered in the field."

#### 2. ########################
print('#### 2. ########################')
sentences_Matheus = sent_tokenize(corpus_Matheus)
print(sentences_Matheus)
print("\nNumber of sentences: " + str(len(sentences_Matheus)))

#### 3. ########################
print('#### 3. ########################')
tokenized_corpus_Matheus = word_tokenize(corpus_Matheus)

#### 4. ########################
print('#### 4. ########################')
wordslist_Matheus = [w for w in tokenized_corpus_Matheus if not w in stopwords.words('english')] 
print(wordslist_Matheus)
print("\nNumber of words: " + str(len(wordslist_Matheus)))
count_vectorizer = CountVectorizer(min_df=2, max_df=20)
ramiro = count_vectorizer.fit_transform(wordslist_Matheus)

#### 5. ########################
print('#### 5. ########################')
print("Size of matrix:" + str(ramiro.shape))

#### 6. ########################
print('#### 6. ########################')
print(count_vectorizer.get_feature_names())

#### 7. ########################
print('#### 7. ########################')
vocabulary = count_vectorizer.get_feature_names()
counts = ramiro.toarray().sum(axis=0)

print("Counts: ")
for i in range(len(vocabulary)):
  print(vocabulary[i] + ": " + str(counts[i]))

#### 8. ########################
print('#### 8. ########################')

