## Assignment 4.1, 4.2 and 4.3

import pandas as pd
import matplotlib.pyplot as plt
import random

print("[Exercise 4.1]")

# Declare file names
file_activity_labels = "UCI HAR Dataset/activity_labels.txt"
file_features = "UCI HAR Dataset/features.txt"
file_testX = "UCI HAR Dataset/test/X_test.txt"
file_testY = "UCI HAR Dataset/test/y_test.txt"
file_trainX = "UCI HAR Dataset/train/X_train.txt"
file_trainY = "UCI HAR Dataset/train/y_train.txt"

# Import data
activity_labels = pd.read_csv(file_activity_labels, delimiter=" ", header=None, names=['id', 'activity'])
features = pd.read_csv(file_features, delimiter=" ", header=None, names=['id', 'feature'])
test_X = pd.read_csv(file_testX, delimiter=" ", header=None, skipinitialspace=True)
test_Y = pd.read_csv(file_testY, delimiter=" ", header=None, skipinitialspace=True)
train_X = pd.read_csv(file_trainX, delimiter=" ", header=None, skipinitialspace=True)
train_Y = pd.read_csv(file_trainY, delimiter=" ", header=None, skipinitialspace=True)

# Add activity name column to labels
train_Y['label'] = train_Y[0].transform(lambda c: activity_labels['activity'][c - 1])
test_Y['label'] = test_Y[0].transform(lambda c: activity_labels['activity'][c - 1])

# Add labels to the measurements
train_X['label'] = train_Y['label']
test_X['label'] = test_Y['label']

# Combine training and test data
X = pd.concat([train_X, test_X], ignore_index=True)
Y = pd.concat([train_Y, test_Y], ignore_index=True)

### 4.1a: dimensionality of the dataset
print("[4.1a: dimensionality]")
print("Training set :", train_X.shape)  # (7352, 561)
print("    Test set :", test_X.shape)  # (2947, 561)
print()

### 4.1b: statistics
print("[4.1b: statistics]")
nFeatures = 6  # amount of features
# Calculate statistics for some features (columns)
for i in range(0, nFeatures):
    print("Feature :", features['feature'][i])
    print("   Mean : %0.3f" % X[i].mean())
    print(" Median : %0.3f" % X[i].median())
    print(" Stddev : %0.3f" % X[i].std())
    print()

### Exercise 4.2
print("[Exercise 4.2]")

### 4.2a: dimensions
print("[4.2a: dimensionality]")
print("Training set :", train_Y.shape)  # (7352, 1)
print("    Test set :", test_Y.shape)  # (2947, 1)
print()

### 4.2b: bar chart
print("[4.2a: bar chart]")
activity_counts = Y[0].value_counts(normalize=False).sort_index()  # count occurrence

# Make plot
plt.figure(figsize=(15, 6))
plt.xlabel("Activity")
plt.ylabel("Counts")
plt.bar(activity_labels['activity'], activity_counts)
plt.savefig("figures/4_2b.png", dpi=300)
print("[4.2a: bar chart saved]")
print()

### 4.3
print("[Exercise 4.3]")
grouped_X = train_X.groupby('label')
random.seed(9001) # set random number seed
plt.figure(figsize=(6,4))

for i in range(0,10):
    feature_id = random.randint(0,features.shape[0])

    plt.clf()
    grouped_X[feature_id].plot.kde()
    plt.title(features['feature'].values[feature_id])
    plt.legend(grouped_X.groups.keys())
    plt.savefig("figures/4.3_%i.png" % feature_id, dpi = 300)

