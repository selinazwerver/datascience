### Contains the models for the predictions

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pyearth

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

# Determine which features can be used to predict
def calc_variance_categorial(df, cols, target, show=False):
    results = []

    for name, col in df[cols].iteritems():
        total_var = 0
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
        results.append([name, len(keys), total_var, total_var / varTarget])

    results = sorted(results, key=lambda x: x[2]) # sort results based on fraction of variance

    if show: # print results
        for feature, nkeys, var, frac in results:
            print(feature, '& %.2f' % frac)

    return results

data = preprocess_data(data)

## Check which columns are numerical/categorial
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
categorial_variance = calc_variance_categorial(data, categorical_cols, target, False)

## Remove data
threshold = 20
operation_groups = data.fillna(-1).groupby('Operatietype')
removed_types = []
for key in operation_groups.groups.keys():
    if len(operation_groups.get_group(key)) < threshold:
        removed_types.append(key)

removed_types.append(-1)  # also remove nan group
for key in removed_types:
    data = data.drop(operation_groups.get_group(key).index)

## Transform categories to numbers to be used in models
for name, nkeys, var, frac in categorial_variance:
    data[name] = data[name].astype('category').cat.codes

data = data[data['Operatieduur'].notna()]  # remove nan surgery durations
all_features = [l[0] for l in categorial_variance]  # list of all features
result = []

## Make models + predictions
for i in range(1,3):
    features = all_features[0:i]
    Y = data['Operatieduur']
    X = data[features]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    # Linear regression
    LR = linear_model.LinearRegression()
    LR.fit(X_train, Y_train)
    LR_predictions = LR.predict(X_test)

    # Multivariate adaptive regression splines
    MARS = pyearth.Earth()
    MARS.fit(X_train, Y_train)
    MARS_predictions = MARS.predict(X_test)

    # Random forest
    RF = RandomForestRegressor(n_estimators=1000, random_state=42)
    RF.fit(X_train, Y_train)
    RF_predictions = RF.predict(X_test)

    # Multilayer perceptron network
    MLP = MLPRegressor(activation='relu')
    MLP.fit(X_train, Y_train)
    MLP_predictions = MLP.predict(X_test)

    # Store results
    result.append([mean_squared_error(Y_test, LR_predictions)**(1/2),   r2_score(Y_test, LR_predictions),
                   mean_squared_error(Y_test, MARS_predictions)**(1/2), r2_score(Y_test, MARS_predictions),
                   mean_squared_error(Y_test, RF_predictions) ** (1/2), r2_score(Y_test, RF_predictions),
                   mean_squared_error(Y_test, MLP_predictions) ** (1 / 2), r2_score(Y_test, MLP_predictions)])