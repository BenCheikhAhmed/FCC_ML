# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models

# StandardScaler: to scale the data
from sklearn.preprocessing import StandardScaler
# RandomOverSampler: to oversample the data
from imblearn.over_sampling import RandomOverSampler

# KNeighborsClassifier: to train a KNN model
from sklearn.neighbors import KNeighborsClassifier
# GaussianNB: to train a Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# LogisticRegression: to train a Logistic Regression model
from sklearn.linear_model import LogisticRegression
# SVC: to train a SVM model
from sklearn.svm import SVC

# classification_report: to evaluate the model
from sklearn.metrics import classification_report

# define the columns of the data
cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
# upload the data
df = pd.read_csv("Data/magic04.data", names = cols)
# show the first 5 rows of the data
df.head()

# convert the class column to binary
df["class"] = (df["class"] == "g").astype(int)
df.head()

# plot the histograms of the features
for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color="blue", label='Gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color="red", label='Hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    #plt.show()

# split the data into train, validation and test sets
# 60% train, 20% validation, 20% test
train , valid , test = np.split(df.sample(frac=1) , [int(0.6*len(df)), int(0.8*len(df))])

# scale the data
def scale_dataset(dataframe , oversample=False):
    # get the features and the target
    X = dataframe[dataframe.columns[0:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    # scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # oversample the data
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    # concatenate the features and the target
    data = np.hstack((X, np.reshape(y,(-1,1))))

    return data , X , y

# scale the datasets
# oversample the train set
train , X_train , y_train = scale_dataset(train , oversample=True)
# do not oversample the validation and test sets
valid , X_valid , y_valid = scale_dataset(valid , oversample=False)
test , X_test , y_test = scale_dataset(test , oversample=False)

# KNN Implementation
# train a KNN model
# use 5 neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# evaluate the model
# predict the validation set
print("KNN")
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Naive Bayes Implementation
# train a Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# evaluate the model
# predict the validation set
print("Naive Bayes")
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Logistic Regression Implementation
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

# evaluate the model
# predict the validation set
print("Logistic Regression")
y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))

# SVM Implementation
svm_model = SVC()
svm_model = svm_model.fit(X_train , y_train)

# evaluate the model
# predict the validation set
print("SVM")
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))