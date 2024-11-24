import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
