import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn

data_file = 'surgical_case_durations.csv'
data = pd.read_csv(data_file, sep=';', encoding='ISO-8859-1')


# Replace values in dataset such that we can work with it
def int_to_str_map(number):
    number = number.split('.')[0]
    string = ""
    mapping = {1: "A", 2: "B", 3: "C", 4:"D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10 : "J", 11: "K", 12: "L",
               13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18 : "R", 19 : "S", 20: "T", 21: "U", 22: "V", 23: "W",
               24: "X", 25 : "Y", 26: "Z"}
    string += str(mapping.get(int(number)))
    return string

def preprocess_data(df):
    df.replace(to_replace=',', value='.', inplace=True, regex=True)  # replace ',' with '.'
    df.replace(to_replace='Onbekend', inplace=True, value=np.nan)  # remove nan
    df.replace(to_replace='Ander specialisme', inplace=True, value=np.nan)  # replace 'other' surgeon
    # Change categories to letters so they will be categories later
    df['Chirurg'] = df['Chirurg'][df['Chirurg'].notna()].apply(lambda x : int_to_str_map(str(x)))
    df['Anesthesioloog'] = df['Anesthesioloog'][df['Anesthesioloog'].notna()].apply(lambda x : int_to_str_map(str(x)))
    df['CCS'] = df['CCS'][df['CCS'].notna()].apply(lambda x: int_to_str_map(str(x)))
    df['NYHA'] = df['NYHA'][df['NYHA'].notna()].apply(lambda x: int_to_str_map(str(x)))
    return df

data = preprocess_data(data)

# Determine the amount of NaN in the dataset
nan_count = []
categories = []

for name, values in data.iteritems():
    nan_count.append((values.isna().sum() / len(values)) * 100)
    categories.append(name)

plt.figure(figsize=(20, 10))
plt.barh(categories, nan_count)
plt.grid()
plt.xlabel('Percentage of NaN in category [%]')
plt.savefig("figures/nan_percentage.png", dpi=300)

# Determine which features can be used to predict
def calc_variance_categorial(df, cols, target, show=True):
    results = []

    for name, col in df[cols].iteritems():
        varTotal = 0
        groups = df.fillna(-1).groupby(name)  # group and replace nan
        keys = groups.groups.keys()
        # keys = keys - [-1]  # ignore nan
        nTarget = len(df[target])
        varTarget = df[target].var()
        if len(keys) < 2:  # group cannot be splitted
            continue

        for key in keys:  # loop through groups
            group = groups.get_group(key)  # get the group
            if len(group) < 2:
                var = 1
            else:
                var = group[target].var()  # calculate variance of the target column for the group
            weight = len(group) / nTarget  # calculate weight of variance
            varTotal += var * weight  # sum variances

        results.append([name, len(keys), varTotal, varTotal / varTarget])
    if show:
        results = sorted(results, key=lambda x: x[2])
        for feature, nkeys, var, frac in results:
            print(feature, '& %.2f' % frac)

    return results

# Check which columns are numerical or categorical
data = data.drop(['Geplande operatieduur', 'Ziekenhuis ligduur', 'IC ligduur'], 1)
column_names = list(data.columns.values)
numerical_cols = []
categorical_cols = []
for col in column_names:
    try:
        data[col] = data[col].apply(lambda x: float(x))
        numerical_cols.append(col)
    except Exception as e:
        categorical_cols.append(col)

target = 'Operatieduur'
categorial_variance = calc_variance_categorial(data, categorical_cols, target)
numerical_corr = data[numerical_cols].corr()

plt.figure(figsize=(10,7))
sn.heatmap(numerical_corr, annot=True)
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=300)
