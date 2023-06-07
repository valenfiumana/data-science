# Supervised learning (classification):

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# Dataset

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"] # Create a list of column names
df = pd.read_csv("../Data/magic04.csv", names=cols)
print(df.head())

df["class"] = (df["class"] == "g").astype(int)
# - df["class"] == "g" checks each value in the "class" column of df to see if it is equal to string "g".
#   The result is a new column with True / False values.
# - (df["class"] == "g").astype(int) converts the True and False values to integers.
#   It replaces True with 1 and False with 0.
# Now the class column will contain 1 or 0 instead of g and h
print(df.head())
print(df.tail())

# Iterate over each feature (excluding the last column "class")
for col in cols[:-1]:
    # Plot histogram for the "gamma" class (now g == True == 1 in column "class")
    plt.hist(df[df["class"]==1][col], color='blue', label='gamma', alpha=0.7, density=True)
    # Plot histogram for the "hadron" class (now h == False == 0 in column "class")
    plt.hist(df[df["class"]==0][col], color='red', label='hadron', alpha=0.7, density=True)

    plt.title(col)
    plt.ylabel("Probability")
    plt.xlabel(col)
    plt.legend()
    #plt.show()

print(df[df.columns[-1]].values)

# Train, validation, test datasets

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))]) # splits dataframe into three parts
# int(0.6*len(df)) == 60% of the length of df == end of the train set == beginning of valid set
# int(0.8*len(df)) == 80% of the length of df == end of the valid set == beginning of test set
# In summary:
# The train set contains the first 60% of the shuffled dataframe.
# The valid set contains the next 20% of the shuffled dataframe.
# The test set contains the remaining portion, which is approximately 20% of the shuffled dataframe.

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values # extract features from df of all columns except the last one
    y = dataframe[dataframe.columns[-1]].values  # extract values of last column, 1 and 0

    scaler = StandardScaler() # to standardize the features.
    X = scaler.fit_transform(X) # scales the features so that they have zero mean and unit variance

    if oversample:
        ros = RandomOverSampler() # used to perform the oversampling
        X, y = ros.fit_resample(X, y)   #  increases the number of samples in the dataset by generating synthetic samples for the minority class,
                                        # aiming to balance the class distribution (if not, there would be 7419 gammas vs 3993 hadron)

    data = np.hstack((X, np.reshape(y, (-1, 1)))) # horizontally stacks the standardized features X and the labels y to create a single numpy array
    # reshape bc X is 2-dim whereas y is 1-dim
    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)


# KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))

# SVM
from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


