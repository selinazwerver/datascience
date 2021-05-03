### Contains the models for the predictions

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import OneHotEncoder
import pyearth

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

    results = sorted(results, key=lambda x: x[2])  # sort results based on fraction of variance

    if show:  # print results
        for feature, nkeys, var, frac in results:
            print(feature, '& %.2f' % frac)

    return results

def onehotencode(data):
    enc = OneHotEncoder()
    enc.fit(data)
    return enc.transform(data).toarray()

data = preprocess_data(data)
# data_for_initial_r2 = data[data['Operatieduur'].notna()]
# data_for_initial_r2 = data_for_initial_r2[data_for_initial_r2['Geplande operatieduur'].notna()]
# r2_original = r2_score(data_for_initial_r2['Geplande operatieduur'], data_for_initial_r2['Operatieduur'])
# print(r2_original)

## Check which columns are numerical/categorial
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
threshold = 61  # determined to have 10 sugery types left : 61
operation_groups = data.fillna(-1).groupby('Operatietype')
removed_types = []
for key in operation_groups.groups.keys():
    if len(operation_groups.get_group(key)) < threshold:
        removed_types.append(key)

removed_types.append(-1)  # also remove nan group
for key in removed_types:
    data = data.drop(operation_groups.get_group(key).index)

# Remove large over/underestimation
data['Difference'] = abs(data['Operatieduur'] - data['Geplande operatieduur'])  # generate difference column
data['Percentual diff'] = (data['Difference']/data['Geplande operatieduur']) * 100
data = data.drop(data[data['Percentual diff'] > 100].index)

## Transform categories to numbers to be used in models
for name in categorical_cols:
    data[name] = data[name].astype('category').cat.codes

data = data[data['Operatieduur'].notna()]  # remove nan surgery durations

all_features = [l[0] for l in categorial_variance]  # list of all features
result = []
options = ['LR', 'MARS', 'DT', 'RT', 'MLP']

# Make models + predictions
nfeatures = 3  # best result : 1
seed = 41  # to make results reproducable
features = all_features[0:nfeatures]  # select right features
features.append('Operatieduur')

# Remove nan groups
for name, values in data[features].iteritems():
    groups = data.groupby(name)
    keys = groups.groups.keys()
    for key in keys:
        if key == -1:  # remove nan group (-1)
            data = data.drop(groups.get_group(key).index)

Y = data['Operatieduur']
X = data[features[0:nfeatures]]
X_mlp = onehotencode(data[features[0:nfeatures]])  # one hot encoding for mlp to use
X_train_mlp, X_test_mlp, Y_train_mlp, Y_test_mlp = train_test_split(X_mlp, Y, test_size=0.20, random_state=seed)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)


## Hyperparameter tuning
## RANDOM FOREST
# n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]  # ntrees
# criterion = ['mse', 'mae']
# max_features = ['auto', 'sqrt', 'log2']  # number of features to consider at every split
# max_depth = [int(x) for x in np.linspace(start=1, stop=100, num=1)]  # maximum number of levels in tree
# max_depth.append(None)
# min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=1)]  # minimum number of samples required to split a node
# min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=10, num=10)]  # minimum number of samples required at each leaf node
# bootstrap = [True, False]  # method of selecting samples for training each tree
#
# # Create the random grid
# rf_parameters = {'n_estimators': n_estimators,
#                  'criterion': criterion,
#                  'max_features': max_features,
#                  'max_depth': max_depth,
#                  'min_samples_split': min_samples_split,
#                  'min_samples_leaf': min_samples_leaf,
#                  'bootstrap': bootstrap,
#                  'random_state': [seed]}
#
# RF = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# grid_search = GridSearchCV(estimator=RF, param_grid=rf_parameters, cv=3, verbose=2, n_jobs=-1)
# grid_search.fit(X_train, Y_train)  # fit the random search model
# print('Best parameters RF:')
# print(grid_search.best_params_)
# RF_predictions = grid_search.best_estimator_.predict(X_test)
# print('MAE RF:', mean_absolute_error(Y_test, RF_predictions))

# {'bootstrap': False,
# 'criterion': 'mae',
# 'max_depth': None,
# 'max_features': 'sqrt',
# 'min_samples_leaf': 2,
# 'min_samples_split': 2,
# 'n_estimators': 178,
# 'random_state': 41}
# MAE RF: 36.01340798491681

## GRADIENT BOOSTEING REGRESSION
# loss = ['ls', 'lad' ,'huber', 'quantile']
# n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
# criterion = ['friedman_mse', 'mse', 'mae']
# min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=1)]
# min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=10)]
# max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=1)]
# max_depth.append(None)
# max_features = ['auto', 'sqrt', 'log2']
#
# gbr_parameters = {'loss' : loss,
#                   'n_estimators' : n_estimators,
#                   'criterion' : criterion,
#                   'min_samples_split' : min_samples_split,
#                   'min_samples_leaf' : min_samples_leaf,
#                   'max_depth' : max_depth,
#                   'max_features' : max_features,
#                   'random_state': [seed]}
#
# GBR = GradientBoostingRegressor()
# GBR_random = GridSearchCV(estimator=GBR, param_grid=gbr_parameters, cv=3, verbose=2, n_jobs=-1)
# GBR_random.fit(X_train, Y_train)  # fit the random search model
# print('Best parameters GBR:')
# print(GBR_random.best_params_)
# GBR_predictions = GBR_random.best_estimator_.predict(X_test)
# print('GBR MAE:', mean_absolute_error(Y_test, GBR_predictions))

# {'criterion': 'mae',
# 'loss': 'huber',
# 'max_depth': None,
# 'max_features': 'sqrt',
# 'min_samples_leaf': 4,
# 'min_samples_split': 2,
# 'n_estimators': 52,
# 'random_state': 41}
# GBR MAE: 35.98213220081576

## MULTILAYER PERCEPTRON
hidden_layer_sizes = [(i,) for i in range(1, 200, 1)]
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']

mlp_parameters = {'hidden_layer_sizes': hidden_layer_sizes,
                  'activation': activation,
                  'solver': solver,
                  'learning_rate': learning_rate,
                  'random_state': [seed]}

MLP = MLPRegressor()
MLP_search = GridSearchCV(estimator=MLP, param_grid=mlp_parameters, cv=3, verbose=2, n_jobs=-1)
MLP_search.fit(X_train_mlp, Y_train_mlp)
print('Best parameters MLP:')
print(MLP_search.best_params_)
MLP_predictions = MLP_search.best_estimator_.predict(X_test_mlp)
print('GBR MAE:', mean_absolute_error(Y_test_mlp, MLP_predictions))

# {'activation': 'tanh', 'hidden_layer_sizes': (49,), 'learning_rate': 'constant', 'random_state': 41, 'solver': 'lbfgs'}
# GBR MAE: 36.14649576034777