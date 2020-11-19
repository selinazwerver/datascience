import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as kurt
from scipy.fftpack import fft
from scipy.signal import butter
from scipy.signal import lfilter

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
print("[4.4a] Dimensionality")
print("Training set :", train_total_acc_x.shape)  # (7352, 128)
print("    Test set :", test_total_acc_x.shape)  # (2947, 128)
print()

### 4.4b: reconstruct original signal
print("[4.4b] Variances")

# Calculate the variances of the accelerations of all 3 axes
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
time_domain_features["label"] = time_domain_features["label"].transform(lambda c: activity_labels['activity'][c - 1])
grouped_features = time_domain_features.groupby("label")
for feature in ["mean", "std", "kurtosis"]:
    plt.figure()
    grouped_features[feature].plot.kde()
    plt.title(feature)
    plt.legend(grouped_features.groups.keys())
    plt.savefig("figures/4.5_%s.png" % feature, dpi=300)

print("[4.5] Plot saved")
print()

### 4.6
print("[Exercise 4.6]")

# Function to calculate the fft
def calculate_fft(x, T):
    # x: list of signal points, T: sample time
    s = np.ceil(np.size(x) / 2)
    y = (2 / np.size(x)) * fft(x)  # normalise to get proper amplitude
    y = y[0:int(s)]  # take half so the negative part is not used
    ym = abs(y)  # magnitude
    f = np.arange(0, s)
    fspacing = 1 / (np.size(x) * T)
    f = fspacing * f  # frequency axis
    return f, ym


### 4.6a: fft of all 6 activities
fs = 50  # sample frequency
T = 1 / fs  # sample time
nsamples = 1000  # amount of samples to transform

# Create dictionary for the activities
raw_signal = pd.DataFrame(raw_signal, columns=["signal", "label"])
raw_signal["label"] = raw_signal["label"].transform(lambda c: activity_labels['activity'][c - 1])
activities_and_signals = raw_signal.groupby("label")["signal"].apply(list).to_dict()

## TODO: change length of signal we want to transform?
for activity, signal in activities_and_signals.items():
    signal = signal[0:nsamples]
    signal = signal - np.mean(signal)
    t = np.linspace(start=0, stop=T * np.size(signal), num=np.size(signal)) # create time axis
    f, y = calculate_fft(signal, T) # fourier transform

    # Plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t, signal)  # original signal
    plt.grid(True)
    plt.title("Original signal activity %s" % activity)
    plt.ylim(-1, 1)

    plt.subplot(1, 2, 2)
    plt.plot(f, y)  # fft
    plt.grid(True)
    plt.title("FFT activity %s" % activity)
    plt.savefig("figures/4.6_%s.png" % activity, dpi=300)

print("[4.6] Plots saved")
print()

### 4.7
print("[Exercise 4.7]")

nsamples = 3000  # amount of samples to transform
fn = fs/2  # nyquist frequency
upper_limit = 3/fn  # upper limit cutoff frequency
lower_limit = 0.6/fn  # lower limit cutoff frequency
filter_order = 4  # order of the filters

## Low-pass filter
b_lp, a_lp = butter(filter_order, upper_limit, btype='lowpass', output='ba')
i = 1

plt.figure(figsize=(15, 6))
for activity, signal in activities_and_signals.items():
    signal = signal[0:nsamples]
    signal = signal - np.mean(signal)
    signal = lfilter(b_lp, a_lp, signal)
    t = np.linspace(start=0, stop=T * np.size(signal), num=np.size(signal)) # create time axis
    f, y = calculate_fft(signal, T) # fourier transform

    # Plot filtered signal
    plt.subplot(6, 2, i)
    plt.grid(True)
    plt.plot(t, signal, label=activity)  # original signal
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    if i == 1: plt.title("Low-pass filtered")
    i +=1

    # Plot frequency spectrum
    plt.subplot(6, 2, i)
    plt.plot(f, y, label=activity)  # fft
    plt.grid(True)
    plt.xlim(0, 3)
    plt.legend(loc="upper right")
    if i == 2: plt.title("FFT")
    i+=1
plt.tight_layout()
plt.savefig("figures/4.7_lowpass.png", dpi=300)
plt.close()

## High-pass filter
b_hp, a_hp = butter(filter_order, lower_limit, btype='highpass', output='ba')
i = 1

plt.figure(figsize=(15, 6))
for activity, signal in activities_and_signals.items():
    signal = signal[0:nsamples]
    signal = signal - np.mean(signal)
    signal = lfilter(b_hp, a_hp, signal)
    t = np.linspace(start=0, stop=T * np.size(signal), num=np.size(signal)) # create time axis
    f, y = calculate_fft(signal, T) # fourier transform

    # Plot filtered signal
    plt.subplot(6, 2, i)
    plt.grid(True)
    plt.plot(t, signal, label=activity)  # original signal
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    if i == 1: plt.title("High-pass filtered")
    i += 1

    # Plot frequency spectrum
    plt.subplot(6, 2, i)
    plt.plot(f, y, label=activity)  # fft
    plt.grid(True)
    plt.xlim(0, 3)
    plt.legend(loc="upper right")
    if i == 2: plt.title("FFT")
    i+=1
plt.tight_layout()
plt.savefig("figures/4.7_highpass.png", dpi=300)
plt.close()

## Band-pass filter
b_bp, a_bp = butter(filter_order, [lower_limit, upper_limit], btype='bandpass', output='ba')
i=1

plt.figure(figsize=(15, 6))
for activity, signal in activities_and_signals.items():
    signal = signal[0:nsamples]
    signal = signal - np.mean(signal)
    signal = lfilter(b_bp, a_bp, signal)
    t = np.linspace(start=0, stop=T * np.size(signal), num=np.size(signal)) # create time axis
    f, y = calculate_fft(signal, T) # fourier transform

    # Plot filtered signal
    plt.subplot(6, 2, i)
    plt.grid(True)
    plt.plot(t, signal, label=activity)  # original signal
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    if i == 1: plt.title("Band-pass filtered")
    i += 1

    # Plot frequency spectrum
    plt.subplot(6, 2, i)
    plt.plot(f, y, label=activity)  # fft
    plt.grid(True)
    plt.xlim(0, 3)
    plt.legend(loc="upper right")
    if i == 2: plt.title("FFT")
    i += 1
plt.tight_layout()
plt.savefig("figures/4.7_bandpass.png", dpi=300)
plt.close()

print("[4.7] Plots saved")
print()
