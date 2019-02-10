# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 04:38:29 2018

@author: Dr. Doofenshmirtz
"""
import numpy as np
import pandas as pd
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.svm import NuSVC
import pickle
import joblib
from sklearn.model_selection import KFold


#from autocorrect import spell


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def scores(confusion):
    tp_m1 = confusion[0][0]
    tp_0 =  confusion[1][1]
    tp_1 = confusion[2][2]
    
    fn_m1 = confusion[1][0]+confusion[2][0]
    fn_0 = confusion[0][1]+confusion[2][1]
    fn_1 = confusion[0][2]+confusion[1][2]
    
    fp_m1 = confusion[0][1]+confusion[0][2]
    fp_0 = confusion[1][0]+confusion[1][2]
    fp_1 = confusion[2][0]+confusion[2][1]
    
    recall_m1=tp_m1/(tp_m1+fn_m1)
    recall_0=tp_0/(tp_0+fn_0)
    recall_1=tp_1/(tp_1+fn_1)
    
    precision_m1=tp_m1/(tp_m1+fp_m1)
    precision_0=tp_0/(tp_0+fp_0)
    precision_1=tp_1/(tp_1+fp_1)
    
    fscore_m1 = (2*precision_m1*recall_m1)/(precision_m1+recall_m1)
    fscore_0 = (2*precision_0*recall_0)/(precision_0+recall_0)
    fscore_1 = (2*precision_1*recall_1)/(precision_1+recall_1)
    
    print ("Positive class")
    print ("Precision: ", precision_1, " Recall: ", recall_1, " F-score:",fscore_1)
    
    print ("Negative class")
    print ("Precision: ", precision_m1, " Recall: ", recall_m1, " F-score:",fscore_m1)
    
    print ("Neutral class")
    print ("Precision: ", precision_0, " Recall: ", recall_0, " F-score:",fscore_0)
    
    

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocessing(arr):
    sentences= []
    tokens=[]
    newtokens=[]
    stop_words=set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    aspect_term = np.array(arr[:, 2])
    aspect_index = np.array(arr[:, 3])
    aspect_index_start=[]
    aspect_index_end=[]
    
    for a in aspect_index:
        start,end = a.split("--")
        aspect_index_start.append(start)
        aspect_index_end.append(end)
    i=0
    sentence = ''
    for d in arr[:,1]:
        temp=int(aspect_index_start[i])
        start = d[:temp]
        start=start.split()
       
        temp=int(aspect_index_end[i])
        end=d[temp:]
        end=end.split()
        
        d=' '.join(start[-6:])+" "+aspect_term[i]+" "+' '.join(end[0:6])
          
    
        d = d.replace('[comma]',' ').replace('\t',' ').replace('\n', ' ').replace('  ', ' ').strip()
        d = strip_punctuation(d)
    
        tokens= word_tokenize(d)
        newtokens=[w for w in tokens if not w in stop_words]
        tagged_tokens = pos_tag(newtokens)
        
        for w in tagged_tokens:
            sentence += lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])).lower()+' '
        sentences.append(sentence)
        sentence = ''
        i=i+1
      

    return sentences
  
    
#model=NuSVC()
svm_model_linear = SVC(kernel = 'linear', C = 0.82) 
f = open('Predictions.txt','w') 


data = pd.read_table('data1_train.csv', sep=",")
arr = np.array(data)
sentences = preprocessing(arr)

vectorizer = TfidfVectorizer(stop_words='english')
#joblib.dump(vectorizer, 'vectroizer.pkl')

#pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

#with open('vectorizer.pkl', 'wb') as fin:
#    pickle.dump(vectorizer, fin)

x = vectorizer.fit_transform(sentences)

#pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    
pos=[]
neg=[]
neu=[]
#compound=[]
    
sid = SentimentIntensityAnalyzer()
for sent in sentences:
    ss = sid.polarity_scores(sent)
    pos.append(ss['pos'])
    neg.append(ss['neg'])
    neu.append(ss['neu'])
    #compound.append(ss['compound'])
        
        
x =hstack((x,np.array(pos)[:,None]))
x =hstack((x,np.array(neg)[:,None]))
x =hstack((x,np.array(neu)[:,None])).tocsr()
#x =hstack((x,np.array(compound)[:,None])).tocsr()
#print (x.shape)

y = np.array(arr[:, -1], dtype=int)
text_id = aspect_term = np.array(arr[:, 0])


kf = KFold(n_splits=10)

#nbfinalconfusion = np.array([[0,0,0],[0,0,0],[0,0,0]])
linearsvmfinalconfusion = np.array([[0,0,0],[0,0,0],[0,0,0]])
#nbaccuracy=0
linearsvmaccuracy=0

for train_index, test_index in kf.split(x):
    
    svm_model_linear = svm_model_linear.fit(x[train_index], y[train_index])
    linearsvmpredictions = svm_model_linear.predict(x[test_index])
    linearsvcconfusion = confusion_matrix(y[test_index], linearsvmpredictions)
    linearsvmaccuracy = linearsvmaccuracy + accuracy_score(y[test_index], linearsvmpredictions)
    linearsvmfinalconfusion = linearsvmfinalconfusion + linearsvcconfusion
    




#print(linearsvmfinalconfusion)
linearsvmaccuracy=linearsvmaccuracy/10
print ("Linear SVM")

print("Confusion matrix using 10 fold cross validation:\n")

scores(linearsvmfinalconfusion)

print("Overall Accuracy: ", linearsvmaccuracy)




x_train, x_test, y_train, y_test, text_id_train, text_id_test = train_test_split(x, y, text_id, test_size=0.2,  shuffle=False)


model = model.fit(x_train,y_train)


y_pred = model.predict(x_test)
nbconfusion = confusion_matrix(y_test, y_pred)
print ("\nConfusion matrix using 80-20 split for training and testing:\n")
newscores = scores(nbconfusion)
accuracy=accuracy_score(y_test, y_pred)
print ("Accuracy:",accuracy)

for i, j in zip(text_id_test, y_pred):
  f.write("%s;;%s\n" % (i,j))

f.close()









