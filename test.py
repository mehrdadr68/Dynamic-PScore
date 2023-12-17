from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pysentimiento import create_analyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
import copy
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import time
start_time = time.time()
def chunk_into_n(lst, n):
  size = ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )
def NormazlieMinMax(temp_list):
    list_normalized = []
    max_list = max(temp_list)
    min_list = min (temp_list)
    m = max_list - min_list
    for i in range (0,len(temp_list)):
        temp = float (temp_list[i] - min_list)
        z = (temp)/float(m)
        list_normalized.append(copy.deepcopy(z))
    return list_normalized


Data_Reviews_ThumbsUp = pd.read_csv('20000Data.csv',usecols=["votes"])
list_ThumbsUp = []
for i in range(0,len(Data_Reviews_ThumbsUp.values)):
    temp = float(Data_Reviews_ThumbsUp.values[i])
    list_ThumbsUp.append(temp)
Data_Reviews = pd.read_csv('20000Data.csv',usecols=["review"])
list_Review = []
for i in range(0,len(Data_Reviews.values)):
    temp = Data_Reviews.values[i][0]
    temp = temp.lower()
    list_Review.append(temp)
#Calc_TF-IDF
#list_TFIDF =[]
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(list_Review)
#temp = X.toarray()
""" for i in range (0,len(temp)):
    sum_tmp = sum(temp[i])
    list_TFIDF.append(copy.deepcopy(sum_tmp)) """
list_NumWords_Review = []
for i in range(0,len(list_Review)):
    NumberofWords = len(list_Review[i].split())
    list_NumWords_Review.append(copy.deepcopy(NumberofWords))
list_normalized_NumWords = NormazlieMinMax(list_NumWords_Review)
robust_scaler = RobustScaler()
robust_scaler.fit(Data_Reviews_ThumbsUp)
list_normalized_thumbsUp = []
# scale all data points using median and IQR
robust_scaled_data = robust_scaler.transform(Data_Reviews_ThumbsUp)
for i in range(0,len(robust_scaled_data)):
    temp = float(robust_scaled_data[i])
    list_normalized_thumbsUp.append(copy.deepcopy(temp))
#Sentiment Analysis
list_Sentiment = []
analyzer = create_analyzer(task="sentiment", lang="en")
for sentence in list_Review:
    temp = analyzer.predict(sentence)
    class_sen = temp.output
    score_sen = temp.probas[class_sen]
    if (class_sen == 'NEG'):
        score_sen = -(score_sen)
    list_Sentiment.append(copy.deepcopy(score_sen))
list_score = []
for i in range(0,len(list_Review)):
    temp_list = []
    NumberofWords = list_normalized_NumWords[i]
    thumbsup = list_normalized_thumbsUp[i]
    sentiment = (list_Sentiment[i])
    scoreR = thumbsup + NumberofWords - (sentiment) 
    temp_list.append(copy.deepcopy(list_Review[i]))
    temp_list.append(copy.deepcopy(scoreR))
    list_score.append(copy.deepcopy(temp_list))
# vectorization of the texts
#n = 10
#temp = [list_score[i:i+n] for i in range(0, len(list_score), n)]
def sortSecond(val):
    return val[1]
list_score.sort(key=sortSecond,reverse=True)
temp = chunk_into_n(list_score,4)
list_result_text = []
list_result_score = []
for i in range (0,4):
    for j in range (0,len(temp[i])):
        list_result_text.append(copy.deepcopy(temp[i][j][0]))
for i in range (0,4):
    for j in range (0,len(temp[i])):
        list_result_score.append(copy.deepcopy(temp[i][j][1]))
df = pd.DataFrame(list_result_text)
df.to_csv('Results_Review.csv')
df = pd.DataFrame(list_result_score)
df.to_csv('Results_Score.csv')
print("--- %s seconds ---" % (time.time() - start_time))
print ("test")


