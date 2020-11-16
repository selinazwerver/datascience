## Assignment 4.4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as kurt

print("[Loading data]")
print()

# Declare file names
file_activity_labels = "UCI HAR Dataset/activity_labels.txt"
file_features = "UCI HAR Dataset/features.txt"

file_train_total_acc_x = "UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt"
file_train_total_acc_y = "UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt"
file_train_total_acc_z = "UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt"

file_test_total_acc_x = "UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt"
file_test_total_acc_y = "UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt"
file_test_total_acc_z = "UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt"

# Import data
activity_labels = pd.read_csv(file_activity_labels, delimiter=" ", header=None, names=['id', 'activity'])
features = pd.read_csv(file_features, delimiter=" ", header=None, names=['id', 'feature'])

train_total_acc_x = pd.read_csv(file_train_total_acc_x, delimiter=" ", header=None, skipinitialspace=True)
train_total_acc_y = pd.read_csv(file_train_total_acc_y, delimiter=" ", header=None, skipinitialspace=True)
train_total_acc_z = pd.read_csv(file_train_total_acc_z, delimiter=" ", header=None, skipinitialspace=True)

test_total_acc_x = pd.read_csv(file_test_total_acc_x, delimiter=" ", header=None, skipinitialspace=True)
test_total_acc_y = pd.read_csv(file_test_total_acc_y, delimiter=" ", header=None, skipinitialspace=True)
test_total_acc_z = pd.read_csv(file_test_total_acc_z, delimiter=" ", header=None, skipinitialspace=True)

# Combine training and test data
total_acc_x = pd.concat([train_total_acc_x, test_total_acc_x])
total_acc_y = pd.concat([train_total_acc_y, test_total_acc_y])
total_acc_z = pd.concat([train_total_acc_z, test_total_acc_z])

### 4.4a: dimensionality
print("[Exercise 4.4]")
print("[4.4a: dimensionality]")
print("Training set :", train_total_acc_x.shape)  # (7352, 128)
print("    Test set :", test_total_acc_x.shape)  # (2947, 128)
print()

### 4.4b: reconstruct original signal
print("[4.4b: variances]")
# Calculate the variances of the accelerations of all 3 axes
# x_var, y_var, z_var = 0, 0, 0
# for i in range(0,total_acc_x.shape[0]):
#     x_var += total_acc_x.iloc[i].var()
#     y_var += total_acc_y.iloc[i].var()
#     z_var += total_acc_z.iloc[i].var()
#
# print("  Variance x : %0.2f" % x_var)
# print("  Variance y : %0.2f" % y_var)
# print("  Variance z : %0.2f" % z_var)

# Same thing as above but takes less time
x_var = total_acc_x.apply(lambda row: row.var(), axis=1).sum()
y_var = total_acc_y.apply(lambda row: row.var(), axis=1).sum()
z_var = total_acc_z.apply(lambda row: row.var(), axis=1).sum()
variances = [x_var, y_var, z_var]
greatest_var = ['x', 'y', 'z'][variances.index(max(variances))]


print("Variance x : %0.2f" % x_var)
print("Variance y : %0.2f" % y_var)
print("Variance z : %0.2f" % z_var)
print("Greatest variance : %s" % greatest_var)
print()

# Select the corresponding body acceleration file
file_test_body_acc = "UCI HAR Dataset/test/Inertial Signals/body_acc_%s_test.txt" % greatest_var
file_train_body_acc = "UCI HAR Dataset/train/Inertial Signals/body_acc_%s_train.txt" % greatest_var
file_train_Y = "UCI HAR Dataset/train/y_train.txt"

train_body_acc = pd.read_csv(file_train_body_acc, delimiter=" ", header=None, skipinitialspace=True)
train_Y = pd.read_csv(file_train_Y, delimiter=" ", header=None, skipinitialspace=True)

data = train_body_acc.loc[:, :63]  # remove half of the columns to solve overlap
raw_data = data.values  # convert dataframe to numpy array
raw_data = raw_data.flatten()  # flatten 2D dataset to 1D array
raw_labels = np.repeat(train_Y.values.flatten(), 64)  # obtain label per datapoint

# Couple the class labels with the raw data points
raw_signal = np.vstack((raw_data, raw_labels)).T

### 4.5a: time domain features
print("[Exercise 4.5]")
# Calculate mean, std, kurtosis per window
mean, std, kurtosis, labels = [], [], [], []
for i in range(0, raw_data.size - 64, 64):
    window = raw_data[i:i + 128]  # select the window
    mean.append(window.mean())
    std.append(window.std())
    kurtosis.append(kurt(window))
    labels.append(raw_labels[i])

### 4.5a: feature plots
# Transform to dataframe for plotting
time_domain_features = pd.DataFrame(np.vstack((mean, std, kurtosis, labels)).T,
                                    columns=["mean", "std", "kurtosis", "label"])
# Transform labels to strings 
time_domain_features["label"] = time_domain_features["label"].transform(lambda c : activity_labels['activity'][c-1])
grouped_features = time_domain_features.groupby("label")
for feature in ["mean", "std", "kurtosis"]:
    plt.figure()
    grouped_features[feature].plot.kde()
    plt.title(feature)
    plt.legend(grouped_features.groups.keys())
    plt.savefig("figures/4.5_%s.png" % feature, dpi=300)

print("[4.4: plot saved]")
print()
