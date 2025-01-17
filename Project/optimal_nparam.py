import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import pyearth
from statsmodels.stats.outliers_influence import variance_inflation_factor


data_file = 'surgical_case_durations.csv'
data = pd.read_csv(data_file, sep=';', encoding='ISO-8859-1')
data_original = data  # might need it later

# Replace values in dataset such that we can work with it
def int_to_str_map(number):
    number = number.split('.')[0]
    string = ""
    mapping = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L",
               13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W",
               24: "X", 25: "Y", 26: "Z"}
    string += str(mapping.get(int(number)))
    return string


def preprocess_data(df):
    df.replace(to_replace=',', value='.', inplace=True, regex=True)  # replace ',' with '.'
    df.replace(to_replace='Onbekend', inplace=True, value=np.nan)  # remove nan
    df.replace(to_replace='Ander specialisme', inplace=True, value=np.nan)  # replace 'other' surgeon
    # Change categories to letters so they will be categories later
    df['Chirurg'] = df['Chirurg'][df['Chirurg'].notna()].apply(lambda x: int_to_str_map(str(x)))
    df['Anesthesioloog'] = df['Anesthesioloog'][df['Anesthesioloog'].notna()].apply(lambda x: int_to_str_map(str(x)))
    df['CCS'] = df['CCS'][df['CCS'].notna()].apply(lambda x: int_to_str_map(str(x)))
    df['NYHA'] = df['NYHA'][df['NYHA'].notna()].apply(lambda x: int_to_str_map(str(x)))
    return df


# Determine which features can be used to predict
def calc_variance_categorial(df, cols, target, show=True):
    results = []

    for name, col in df[cols].iteritems():
        total_var = 0
        total_size = 0
        groups = df.fillna(-1).groupby(name)  # replace nan and group
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
            total_var += var * weight  # sum variances
        missing_percentage = len(groups.get_group(-1))/len(col)
        results.append([name, total_var, total_var / varTarget, missing_percentage*100,
                        (total_var/varTarget)/(1-missing_percentage)])

    results = sorted(results, key=lambda x: x[4]) # sort results based on fraction of variance

    return results

def onehotencode(data):
    enc = OneHotEncoder()
    enc.fit(data)
    return enc.transform(data).toarray()

data = preprocess_data(data)
data_for_initial_mse = data[data['Operatieduur'].notna()]
data_for_initial_mse = data_for_initial_mse[data_for_initial_mse['Geplande operatieduur'].notna()]
mse_original = mean_squared_error(data_for_initial_mse['Geplande operatieduur'], data_for_initial_mse['Operatieduur'])

## Check which columns are numerical/categorial
data_temp = data.drop(['Geplande operatieduur', 'Ziekenhuis ligduur', 'IC ligduur'], 1)
column_names = list(data_temp.columns.values)
numerical_cols = []
categorical_cols = []
for col in column_names:
    try:
        data_temp[col] = data_temp[col].apply(lambda x: float(x))
        numerical_cols.append(col)
    except Exception as e:
        categorical_cols.append(col)
target = 'Operatieduur'
# categorial_variance = calc_variance_categorial(data_temp, categorical_cols, target, False)

## Remove data
threshold = 61  # determined to have 10 surgery types left : 61
operation_groups = data.fillna(-1).groupby('Operatietype')
removed_types = []
for key in operation_groups.groups.keys():
    if len(operation_groups.get_group(key)) < threshold:
        removed_types.append(key)

removed_types.append(-1)  # also remove nan group
for key in removed_types:
    data = data.drop(operation_groups.get_group(key).index)

## Transform categories to numbers to be used in models
for name in categorical_cols:
    data[name] = data[name].astype('category').cat.codes


data = data[data['Operatieduur'].notna()]  # remove nan surgery durations

all_features = categorical_cols  # list of all features

## Store results
result = []
result_LR = []
result_MARS = []
result_RF = []
result_MLP = []
result_GBR = []

# Make models + predictions
seed = 41  # to make results reproducable
remaining_data = []

for nfeatures in range(1, len(all_features)):

    features = all_features[0:nfeatures]  # select right features
    features.append('Operatieduur')

    # Remove nan groups
    data_clean = data
    for name, values in data_clean[features].iteritems():
        data_clean = data_clean.fillna(-1)
        groups = data_clean.groupby(name)
        keys = groups.groups.keys()
        for key in keys:
            if key == -1:  # remove nan group (-1)
                data_clean = data_clean.drop(groups.get_group(key).index)

    data_clean['Difference'] = abs(data['Operatieduur'] - data['Geplande operatieduur'])  # generate difference column
    data_clean['Percentual diff'] = (data_clean['Difference'] / data['Geplande operatieduur']) * 100
    data_clean = data_clean.drop(data_clean[data_clean['Percentual diff'] > 100].index)

    Y = data['Operatieduur']
    X = data[features[0:nfeatures]]
    X_mlp = onehotencode(data[features[0:nfeatures]])  # one hot encoding for mlp to use
    X_train_mlp, X_test_mlp, Y_train_mlp, Y_test_mlp = train_test_split(X_mlp, Y, test_size=0.20, random_state=seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

    remaining_data.append([nfeatures, len(data_clean.groupby('Operatietype').groups.keys()), len(Y_train), len(Y_test)])

    # Linear regression
    LR = LinearRegression()
    LR.fit(X_train_mlp, Y_train_mlp)
    LR_predictions = LR.predict(X_test_mlp)
    result_LR.append(mean_absolute_error(Y_test_mlp, LR_predictions))

    # Multivariate adaptive regression splines
    MARS = pyearth.Earth()
    MARS.fit(X_train_mlp, Y_train_mlp)
    MARS_predictions = MARS.predict(X_test_mlp)
    result_MARS.append(mean_absolute_error(Y_test_mlp, MARS_predictions))

    # print(nfeatures)
    # print('LR:', mean_absolute_error(Y_test_mlp, LR_predictions))
    # print('MARS:', mean_absolute_error(Y_test_mlp, MARS_predictions))
    # print()
    # if nfeatures == 4: exit()

    # Random forest
    RF = RandomForestRegressor(random_state=seed)
    RF.fit(X_train, Y_train)
    RF_predictions = RF.predict(X_test)
    result_RF.append(mean_absolute_error(Y_test, RF_predictions))

    # Multilayer perceptron network
    MLP = MLPRegressor()
    MLP.fit(X_train_mlp, Y_train_mlp)
    MLP_predictions = MLP.predict(X_test_mlp)
    result_MLP.append(mean_absolute_error(Y_test_mlp, MLP_predictions))

    # Gradient booster regression
    GBR = GradientBoostingRegressor(random_state=seed)
    GBR.fit(X_train, Y_train)
    GBR_predictions = GBR.predict(X_test)
    result_GBR.append(mean_absolute_error(Y_test, GBR_predictions))

plt.figure()
plt.plot(range(1, len(all_features)), result_LR, label='LR', color='orangered')
plt.plot(range(1, len(all_features)), result_MARS, label='MARS', color='mediumvioletred')
plt.plot(range(1, len(all_features)), result_RF, label='RF', color='indigo')
plt.plot(range(1, len(all_features)), result_MLP, label='MLP', color='coral')
plt.plot(range(1, len(all_features)), result_GBR, label='GBR', color='plum')
plt.ylim([33, 45])
plt.legend()
plt.xlabel('Amount of features')
plt.ylabel('MAE')
plt.grid()
plt.savefig('figures/mae_per_feature_amount.png', dpi=300)
# plt.show()

print('#f', '#o'.rjust(3), 'train'.rjust(6), 'test')
print(tabulate(remaining_data))
print()
print('#f', 'LR'.rjust(3), 'MARS'.rjust(10), 'RF'.rjust(6), 'MLP'.rjust(9), 'GBR'.rjust(8))
print(tabulate(np.array([range(1,len(all_features)), result_LR, result_MARS, result_RF, result_MLP, result_GBR]).T))