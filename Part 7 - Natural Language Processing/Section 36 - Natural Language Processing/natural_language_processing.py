# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# -----------------Homework---------------------------

## 1. Run the other classification models we made in Part 3 - Classification, other than the one we used in the last tutorial.
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(X_train, y_train)

# Predicting the Test set results
lr_y_pred = lr_classifier.predict(X_test)

# Making the Confusion Matrix
lr_cm = confusion_matrix(y_test, lr_y_pred)

## 2. Evaluate the performance of each of these models.
nb_accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
nb_precision = cm[0][0] / (cm[0][0] + cm[0][1])
nb_recall = cm[0][0] / (cm[0][0] + cm[1][0])
nb_f1_score = (2 * nb_precision * nb_recall) / (nb_precision + nb_recall) 

lr_accuracy = (lr_cm[0][0] + lr_cm[1][1]) / lr_cm.sum()
lr_precision = lr_cm[0][0] / (lr_cm[0][0] + lr_cm[0][1])
lr_recall = lr_cm[0][0] / (lr_cm[0][0] + lr_cm[1][0])
lr_f1_score = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)

## 3. Try even other classification models that we haven't covered in Part 3 - Classification.
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=0, max_features = 1500)
dt_classifier.fit(X_train, y_train)

# Predicting the Test set results
dt_y_pred = dt_classifier.predict(X_test)

# Making the Confusion Matrix
dt_cm = confusion_matrix(y_test, dt_y_pred)

dt_accuracy = (dt_cm[0][0] + dt_cm[1][1]) / dt_cm.sum()
dt_precision = dt_cm[0][0] / (dt_cm[0][0] + dt_cm[0][1])
dt_recall = dt_cm[0][0] / (dt_cm[0][0] + dt_cm[1][0])
dt_f1_score = (2 * dt_precision * dt_recall) / (dt_precision + dt_recall)


