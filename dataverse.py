# importing Module

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


import pickle


# preprocessing text

tr = pd.read_csv('/content/drive/MyDrive/data_vrese/train.csv')
tr = tr.dropna()
# Create a list of the categories to remove
categories_to_remove = ["q-alg", "funct-an", "alg-geom"]

# Get the indices of the rows that contain the categories to remove
indices_to_remove = tr[tr["category"].isin(categories_to_remove)].index

# Remove the rows at the specified indices
tr = tr.drop(indices_to_remove)
tr = tr.drop_duplicates(subset=["abstract"],keep='first')
te = pd.read_csv('/content/drive/MyDrive/data_vrese/test.csv')

# concat both frame

df = pd.concat([tr,te],ignore_index=True)

df.head()

# spliting into train test

x_train, y_train = df['abstract'],df['category']
x_test, y_test = te['abstract'],te['category']

# label encoding

label_encoder = LabelEncoder().fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# text encoding

vector = CountVectorizer(max_features = 100000, ngram_range = (1,2),stop_words = 'english')
x_train = vector.fit_transform(x_train)
x_test = vector.transform(x_test)


# modeling

from sklearn.svm import LinearSVC
clf=LinearSVC(random_state=0)
clf.fit(x_train, y_train)    # training model on train data


clfval = clf.predict(x_test)   # predicting val data
print('F1 Score : {}'.format(f1_score(y_test, clfval, average='micro')))  # printing F1 score



# inverse label encodingg

y_pred = label_encoder.inverse_transform(clfval)

# making submission


sub = pd.DataFrame()
sub['id'] = te.id
sub['category'] = y_pred
sub.to_csv("submission.csv", index=False)

# saving model


print('SAVING TFID MODEL FOR FUTURE USE!!')

# saving tfid model for further use
with open('CountVectorizer100000.pkl', 'wb') as file:
    pickle.dump(vector, file)


print('SAVED TFID MODEL!!')

# saving y_label model for further use


print('SAVING LABELING MODEL FOR FUTURE USE!!')


with open('labelencoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print('SAVED LABEL ENCODING MODEL!!')

print('saving model for future')



with open('model_svc.pkl', 'wb') as file:
    pickle.dump(clf, file)


print('model saved')
