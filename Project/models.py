### Contains the models for the predictions

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tabulate import tabulate
import statistics as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import pyearth
import matplotlib.pyplot as plt

data_file = 'surgical_case_durations.csv'
data = pd.read_csv(data_file, sep=';', encoding='ISO-8859-1')
data_original = data  # might need it later

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calc_improvement(real, model, option='percent'):
    if option == 'percent':
        return ((real - model)/real)*100
    if option == 'minute':
        return (model-real)

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
order = []  # to store which number corresponds to which operation type
for name in categorical_cols:
    data[name] = data[name].astype('category').cat.codes
data = data[data['Operatieduur'].notna()]  # remove nan surgery durations

all_features = [l[0] for l in categorial_variance]  # list of all features
result = []
options = ['LR', 'MARS', 'DT', 'RT', 'MLP']

# Make models + predictions
nfeatures = 3  # amount of features to take into accoutn
seed = 41  # to make results reproducable
features = all_features[0:nfeatures]  # select right features
features.append('Operatieduur')

# Remove nan groups
data = data.fillna(-1)
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
baseline = data.loc[X_test.index.values, ['Geplande operatieduur']]

# Linear regression
LR = LinearRegression()
LR.fit(X_train_mlp, Y_train_mlp)
LR_predictions = LR.predict(X_test_mlp)
result.append(['LR', mean_absolute_error(Y_test_mlp, LR_predictions),
               mean_absolute_percentage_error(Y_test_mlp, LR_predictions),
               mean_squared_error(Y_test_mlp, LR_predictions) ** (1 / 2),
               # r2_score(Y_test, LR_predictions),
               calc_improvement(mean_absolute_error(Y_test_mlp, baseline),
                                mean_absolute_error(Y_test_mlp, LR_predictions))])

# Multivariate adaptive regression splines
MARS = pyearth.Earth()
MARS.fit(X_train_mlp, Y_train_mlp)
MARS_predictions = MARS.predict(X_test_mlp)
result.append(['MARS', mean_absolute_error(Y_test_mlp, MARS_predictions),
               mean_absolute_percentage_error(Y_test_mlp, MARS_predictions),
               mean_squared_error(Y_test_mlp, MARS_predictions) ** (1 / 2),
               # r2_score(Y_test, MARS_predictions),
               calc_improvement(mean_absolute_error(Y_test_mlp, baseline), mean_absolute_error(Y_test, MARS_predictions))])

# Random forest
RF = RandomForestRegressor(n_estimators=178, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',
                           max_depth=None, bootstrap=False, criterion='mae', random_state=seed)

RF.fit(X_train, Y_train)
RF_predictions = RF.predict(X_test)
result.append(['RF', mean_absolute_error(Y_test, RF_predictions),
               mean_absolute_percentage_error(Y_test, RF_predictions),
               mean_squared_error(Y_test, RF_predictions) ** (1 / 2),
               # r2_score(Y_test, RF_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, RF_predictions))])

# Multilayer perceptron network
MLP = MLPRegressor(activation='tanh', solver='lbfgs', hidden_layer_sizes=(49,),
                   learning_rate='constant', random_state=seed)
MLP.fit(X_train_mlp, Y_train_mlp)
MLP_predictions = MLP.predict(X_test_mlp)
result.append(['MLP', mean_absolute_error(Y_test_mlp, MLP_predictions),
               mean_absolute_percentage_error(Y_test_mlp, MLP_predictions),
               mean_squared_error(Y_test_mlp, MLP_predictions) ** (1 / 2),
               # r2_score(Y_test, MLP_predictions),
               calc_improvement(mean_absolute_error(Y_test_mlp, baseline), mean_absolute_error(Y_test_mlp, MLP_predictions))])

# Gradient boosting regression
GBR = GradientBoostingRegressor(n_estimators=52, min_samples_split=2, min_samples_leaf=4, max_features='sqrt',
                                max_depth=None, loss='huber', criterion='mae', random_state=seed)
GBR.fit(X_train, Y_train)
GBR_predictions = GBR.predict(X_test)
result.append(['GBR', mean_absolute_error(Y_test, GBR_predictions),
               mean_absolute_percentage_error(Y_test, GBR_predictions),
               mean_squared_error(Y_test, GBR_predictions) ** (1 / 2),
               # r2_score(Y_test, GBR_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, GBR_predictions))])

print('MAE'.rjust(9), 'MAPE'.rjust(9), 'RMSE'.rjust(8), 'Improvement'.rjust(15))
print(tabulate(result))

#       MAE      MAPE     RMSE     Improvement
# ----  -------  -------  -------  -------
# LR    36.2834  21.6062  48.745   7.33727
# MARS  36.386   21.7019  48.7831  7.07534
# RF    36.01    21.5393  48.4642  8.03573
# MLP   36.3995  21.8516  48.8937  7.04097
# GBR   36.0277  21.6406  48.4901  7.99039
# ----  -------  -------  -------  -------

# Determine statistics per operation type
test_individual = pd.concat([X_test[features[0:nfeatures]], Y_test, baseline], axis=1)
test_individual['New index'] = range(0, len(test_individual))
test_individual = test_individual.set_index('New index')

order = ['AVR', 'AVR + MVP shaving', 'CABG', 'CABG + AVR', 'CABG + Pacemakerdraad tijdelijk',
         'Lobectomie of segmentresectie', 'Mediastinoscopie', 'MVP', 'Rethoracotomie', 'Wondtoilet']
i = 0
results_per_operation_LR = []
results_per_operation_MARS = []
results_per_operation_RF = []
results_per_operation_MLP = []
results_per_operation_GBR = []
mean_duration = []
nobservations = []
average_duration = []

groups = test_individual.groupby(['Operatietype'])
keys = groups.groups.keys()

for key in keys:
    group = groups.get_group(key)
    X = pd.DataFrame(group[features[0:nfeatures]])
    X_mlp = X_test_mlp[X.index]

    Y = pd.DataFrame(group['Operatieduur'])
    Y_list = Y.values

    initial = pd.DataFrame(group['Geplande operatieduur'])
    nobservations.append([order[i], len(group)])  # amount of observations
    average_duration.append([order[i], stats.mean(Y_list.flatten()), stats.stdev(Y_list.flatten())])

    initial_error = mean_absolute_error(Y, initial)
    print('Initial error of', order[i], ':', initial_error)

    print(order[i], len(group))

    pred_LR  = LR.predict(X_mlp)
    mae_LR   = mean_absolute_error(Y, pred_LR)
    mape_LR  = mean_absolute_percentage_error(Y, pred_LR)
    # r2_LR    = r2_score(Y, pred_LR)
    imp_LR   = calc_improvement(initial_error, mae_LR, 'minute')
    error_LR = [abs(Y_list[i][0] - pred_LR[i]) for i in range(0,len(pred_LR))]
    std_LR   = stats.stdev(error_LR)
    results_per_operation_LR.append([order[i], mae_LR, mape_LR, imp_LR, std_LR])

    pred_MARS  = MARS.predict(X_mlp)
    mae_MARS   = mean_absolute_error(Y, pred_MARS)
    mape_MARS  = mean_absolute_percentage_error(Y, pred_MARS)
    # r2_MARS    = r2_score(Y, pred_MARS)
    imp_MARS   = calc_improvement(initial_error, mae_MARS, 'minute')
    error_MARS = [abs(Y_list[i][0] - pred_MARS[i]) for i in range(0,len(pred_MARS))]
    std_MARS   = stats.stdev(error_MARS)
    results_per_operation_MARS.append([order[i], mae_MARS, mape_MARS, imp_MARS, std_MARS])

    pred_RF  = RF.predict(X)
    mae_RF   = mean_absolute_error(Y, pred_RF)
    mape_RF  = mean_absolute_percentage_error(Y, pred_RF)
    # r2_RF    = r2_score(Y, pred_RF)
    imp_RF   = calc_improvement(initial_error, mae_RF, 'minute')
    error_RF = [abs(Y_list[i][0] - pred_RF[i]) for i in range(0,len(pred_RF))]
    std_RF   = stats.stdev(error_RF)
    results_per_operation_RF.append([order[i], mae_RF, mape_RF, imp_RF, std_RF])

    pred_MLP  = MLP.predict(X_mlp)
    mae_MLP   = mean_absolute_error(Y, pred_MLP)
    mape_MLP  = mean_absolute_percentage_error(Y, pred_MLP)
    # r2_MLP    = r2_score(Y, pred_MLP)
    imp_MLP   = calc_improvement(initial_error, mae_MLP, 'minute')
    error_MLP = [abs(Y_list[i][0] - pred_MLP[i]) for i in range(0,len(pred_MLP))]
    std_MLP   = stats.stdev(error_MLP)
    results_per_operation_MLP.append([order[i], mae_MLP, mape_MLP, imp_MLP, std_MLP])

    pred_GBR  = GBR.predict(X)
    mae_GBR   = mean_absolute_error(Y, pred_GBR)
    mape_GBR  = mean_absolute_percentage_error(Y, pred_GBR)
    # r2_GBR    = r2_score(Y, pred_GBR)
    imp_GBR   = calc_improvement(initial_error, mae_GBR, 'minute')
    error_GBR = [abs(Y_list[i][0] - pred_GBR[i]) for i in range(0,len(pred_GBR))]
    std_GBR   = stats.stdev(error_GBR)
    results_per_operation_GBR.append([order[i], mae_GBR, mape_GBR, imp_GBR, std_GBR])
    i += 1

print()
print("Amount of observations per surgery in the test set")
print(tabulate(nobservations))
# -----------------------------  ---
# AVR                             75
# AVR + MVP shaving               14
# CABG                           223
# CABG + AVR                      36
# CABG + pacemaker tijdelijk      69
# Lobectomie of segmentresectie    2
# Mediastinoscopie                20
# MVB                              2
# -----------------------------  ---

print()
print("Mean + std of each surgery type")
print(tabulate(average_duration))

# -------------------------------  -------  -------
# AVR                              205.133  44.4578
# AVR + MVP shaving                236.643  34.6557
# CABG                             230.695  48.6179
# CABG + AVR                       283.167  64.3013
# CABG + Pacemakerdraad tijdelijk  263.101  46.6987
# Lobectomie of segmentresectie    247.5    38.8909
# Mediastinoscopie                 224.05   68.9336
# MVP                               60.5    26.163
# -------------------------------  -------  -------

print()
print("LR")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(10), "Imp".rjust(8), "Std".rjust(11))
print(tabulate(results_per_operation_LR))

print()
print("MARS")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(9), "Imp".rjust(8), "Std".rjust(10))
print(tabulate(results_per_operation_MARS))

print()
print("RF")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(9), "Imp".rjust(8), "Std".rjust(11))
print(tabulate(results_per_operation_RF))

print()
print("MLP")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(9), "Imp".rjust(8), "Std".rjust(11))
print(tabulate(results_per_operation_MLP))

print()
print("GBR")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(9), "Imp".rjust(8), "Std".rjust(11))
print(tabulate(results_per_operation_GBR))

# LR
# Surgery type                     MAE       MAPE     Imp        Std
# -------------------------------  --------  -------  ---------  --------
# AVR                               35.9467  22.5247    4.97333  30.4133
# AVR + MVP shaving                 31.7857  13.4262    6.57143  24.792
# CABG                              33.9686  24.6324   -2.43498  30.0199
# CABG + AVR                        48.6667  17.318   -20.4167   44.1653
# CABG + Pacemakerdraad tijdelijk   33.4638  16.8601   -7.76812  30.4885
# Lobectomie of segmentresectie    109       43.3409   17.5      24.0416
# Mediastinoscopie                  48.45    40.8979    2.85     44.1224
# MVP                               18.5     34.569    -5         2.12132
# -------------------------------  --------  -------  ---------  --------
#
# MARS
# Surgery type                   MAE      MAPE      Imp        Std
# -------------------------------  -------  -------  ---------  --------
# AVR                              36.6365  22.79      5.66315  30.8549
# AVR + MVP shaving                31.1484  13.4319    5.93413  24.9687
# CABG                             34.015   24.8685   -2.3886   30.1854
# CABG + AVR                       49.7533  17.3537  -19.3301   44.4715
# CABG + Pacemakerdraad tijdelijk  33.7199  17.437    -7.51196  30.6771
# Lobectomie of segmentresectie    76.5875  30.0813  -14.9125    1.41524
# Mediastinoscopie                 48.4553  40.5763    2.85534  43.5219
# MVP                              18.5     27.5995   -5        15.5563
# -------------------------------  -------  -------  ---------  --------
#
# RF
# Surgery type                   MAE      MAPE      Imp         Std
# -------------------------------  -------  -------  ---------  --------
# AVR                              34.2482  21.693     3.27487  29.9559
# AVR + MVP shaving                32.1106  13.4795    6.89627  25.3002
# CABG                             34.2837  25.04     -2.1199   30.4274
# CABG + AVR                       49.7105  17.1828  -19.3728   44.4776
# CABG + Pacemakerdraad tijdelijk  33.3405  16.3474   -7.89139  30.5624
# Lobectomie of segmentresectie    60.4466  23.4782  -31.0534   33.5955
# Mediastinoscopie                 48.45    41.5716    2.85     43.8603
# MVP                              18.5     34.6912   -5         2.43118
# -------------------------------  -------  -------  ---------  --------
#
# MLP
# Surgery type                   MAE      MAPE      Imp         Std
# -------------------------------  --------  -------  ---------  -------
# AVR                               35.2045  22.3058    4.23121  30.2428
# AVR + MVP shaving                 31.7484  13.2144    6.53413  22.9323
# CABG                              34.2452  24.9685   -2.15836  30.3356
# CABG + AVR                        48.5098  17.4296  -20.5735   43.5779
# CABG + Pacemakerdraad tijdelijk   33.3331  16.6859   -7.89881  30.4767
# Lobectomie of segmentresectie    102.164   40.5444   10.664    38.8898
# Mediastinoscopie                  52.1499  40.4637    6.54986  44.7111
# MVP                               18.5     41.6086   -5        19.9767
# -------------------------------  --------  -------  ---------  -------
#
# GBR
# Surgery type                   MAE      MAPE      Imp         Std
# -------------------------------  -------  -------  ---------  --------
# AVR                              34.4062  21.871     3.43285  30.0615
# AVR + MVP shaving                31.8084  13.3678    6.59412  24.2442
# CABG                             34.3319  25.152    -2.07168  30.5673
# CABG + AVR                       48.899   17.1676  -20.1843   44.3774
# CABG + Pacemakerdraad tijdelijk  33.3231  16.6623   -7.90878  30.4802
# Lobectomie of segmentresectie    70.3814  27.5424  -21.1186   33.1435
# Mediastinoscopie                 48.45    41.6988    2.85     43.3673
# MVP                              18.5     32.5251   -5         3.06293
# -------------------------------  -------  -------  ---------  --------


