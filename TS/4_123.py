## Assignment 4.1

import pandas as pd

print("[Exercise 4.1]")

# Declare file names
file_activity_labels = "/home/selina/Desktop/DS/datascience/TS/UCI_HAR_Dataset/activity_labels.txt"
file_features = "/home/selina/Desktop/DS/datascience/TS/UCI_HAR_Dataset/features.txt"
file_testX = "/home/selina/Desktop/DS/datascience/TS/UCI_HAR_Dataset/test/X_test.txt"
file_testY = "/home/selina/Desktop/DS/datascience/TS/UCI_HAR_Dataset/test/y_test.txt"
file_trainX = "/home/selina/Desktop/DS/datascience/TS/UCI_HAR_Dataset/train/X_train.txt"
file_trainY = "/home/selina/Desktop/DS/datascience/TS/UCI_HAR_Dataset/train/y_train.txt"

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

X = pd.concat([train_X, test_X], ignore_index=True)
Y = pd.concat([train_Y, test_Y], ignore_index=True)

### 4.1a: dimensionality of the dataset
print("[4.1a: dimensionality]")
print("Training set :", train_X.shape)  # (7352, 561)
print("    Test set :", test_X.shape)  # (2947, 561)
print()

### 4.1b: statistics
print("[4.1b: statistics]")
nFeatures = 5 # amount of features

for i in range(0, nFeatures):
    print(" Feature :", features['feature'][i])
    print("    Mean : %0.3f" % X.head(nFeatures)[i].mean())
    print("  Median : %0.3f" % X.head(nFeatures)[i].median())
    print("  Stddev : %0.3f" % X.head(nFeatures)[i].std())
    print()
