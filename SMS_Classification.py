
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import unicodedata
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
import time

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[2]:


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith('P')
)


# In[3]:


def load_sms_data(path):
    return pd.read_csv(path, encoding='utf-8', sep=',', names=["class", "msg"])


# In[4]:


def remove_punctuation(text):
    return text.translate(tbl)


# In[5]:


def clean_sms(sms_message):
    return remove_punctuation(sms_message).lower()


# In[6]:


def class_str_to_int(class_str):
    if class_str == "ham":
        return -1
    elif class_str == "spam":
        return 1
    else:
        return None


# In[7]:


def accuracy(preds, targets):
    return ((preds == targets).sum())/len(targets)


# In[8]:


sms_data_df = load_sms_data("Data/sms/smsspamcollection/sms_spam3.csv")


# In[9]:


sms_data_df


# In[10]:


# clean messages
sms_data_df['clean_msg'] = sms_data_df['msg'].astype('str').apply(clean_sms)
sms_data_df['label'] = sms_data_df['class'].astype(
    'str').apply(class_str_to_int)

sms_data_df


# In[11]:


email_data_df = pd.read_pickle('email_cleaned')

x = email_data_df['clean_msg']
y = email_data_df['target']


# In[12]:


np.random.seed(int(time.time()))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8)


# In[13]:


print("Train set: %s" % len(x_train))
print("Test set: %s" % len(x_test))


# In[14]:


vectorizer = CountVectorizer()

train_matrix = vectorizer.fit_transform(x_train)
test_matrix = vectorizer.transform(x_test)


# In[15]:


#print(vectorizer.get_feature_names())
print(len(vectorizer.get_feature_names()))


# In[16]:


print(train_matrix.shape, test_matrix.shape)


# In[17]:


def perform_comparison(classifiers):
    res = dict()
    for model_data in classifiers:
        name = model_data['name']
        model = model_data['model']
        
        model.fit(train_matrix, y_train)
        y_test_pred = model.predict(test_matrix)
        y_train_pred = model.predict(train_matrix)
        res[name] = accuracy(y_train_pred, y_train), accuracy(y_test_pred, y_test)
        
    return res


# In[ ]:


classifiers = [
    {
        'name': "Nearest Neighbors",
        'model': KNeighborsClassifier(3)
    },
    {
        'name': "Linear SVM",
        'model': SVC(kernel="linear", C=0.025)
    },
    {
        'name': "Decision Tree",
        'model': DecisionTreeClassifier(max_depth=5)
    },
    {
        'name': "Random Forest",
        'model': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    },
    {
        'name': "Naive Bayes",
        'model': MultinomialNB()
    }
]

result = perform_comparison(classifiers)
for name in result.keys():
    print("Model name: %s" % name)
    print("Accuracy for train set:\t%f" % result[name])
    print("Accuracy for test set:\t%f <<" % result[name])
    print()

