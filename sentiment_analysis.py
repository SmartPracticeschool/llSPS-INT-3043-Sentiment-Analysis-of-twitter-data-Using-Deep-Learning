# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:45:38 2020

@author: tewar
"""

# Importing libraries
import numpy as np
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('train.csv', encoding='ISO-8859-1')

dataset2 = dataset.iloc[:2000,:]

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]

def remove_pat(inpt, pat): #for removing the @names and links in the comments
    n = re.findall(pat, inpt)
    for i in n:
        inpt = re.sub(i, ' ', inpt)
    return inpt

for i in range(0, 2000):
    rev = dataset2['SentimentText'][i]
    rev = remove_pat(rev, '@[\w]*') #user name
    rev = rev.replace('(', '') #bracket one
    rev = rev.replace(')', '') #bracket two
    rev = remove_pat(rev, r'https?://[A-Za-z0-9./]+')
    #rev = remove_pat(rev, r"http\S+") #links
    rev = re.sub('[^a-zA-Z]', ' ',rev) #removing special characters
    rev = rev.lower() #lower case
    rev = rev.split()
    rev = [word for word in rev if not word in set(stopwords.words('english'))] #getting rid of stopwords
    ps = PorterStemmer()
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))] #Stemming
    rev = ' '.join(rev)
    c.append(rev)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3800)
x = cv.fit_transform(c).toarray()
y = dataset2.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)
