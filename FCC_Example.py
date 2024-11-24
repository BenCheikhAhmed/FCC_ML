import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

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
    plt.show()

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