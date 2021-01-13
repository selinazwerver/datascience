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
from sklearn.preprocessing import StandardScaler
import pyearth

data_file = 'surgical_case_durations.csv'
data = pd.read_csv(data_file, sep=';', encoding='ISO-8859-1')
data_original = data  # might need it later

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calc_improvement(real, model):
    return ((real - model)/real)*100

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

data = preprocess_data(data)
# data_for_initial_r2 = data[data['Operatieduur'].notna()]
# data_for_initial_r2 = data_for_initial_r2[data_for_initial_r2['Geplande operatieduur'].notna()]
# r2_original = r2_score(data_for_initial_r2['Geplande operatieduur'], data_for_initial_r2['Operatieduur'])
# print('R2 of scheduled and actual surgery duration:', r2_original)
# print()

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
nfeatures = 23  # best result : 1
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
baseline = data.loc[X_test.index.values, ['Geplande operatieduur']]

# Linear regression
LR = LinearRegression()
LR.fit(X_train, Y_train)
LR_predictions = LR.predict(X_test)
result.append(['LR', mean_absolute_error(Y_test, LR_predictions),
               mean_absolute_percentage_error(Y_test, LR_predictions),
               mean_squared_error(Y_test, LR_predictions) ** (1 / 2),
               # r2_score(Y_test, LR_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, LR_predictions))])

# Multivariate adaptive regression splines
MARS = pyearth.Earth()
MARS.fit(X_train, Y_train)
MARS_predictions = MARS.predict(X_test)
result.append(['MARS', mean_absolute_error(Y_test, MARS_predictions),
               mean_absolute_percentage_error(Y_test, MARS_predictions),
               mean_squared_error(Y_test, MARS_predictions) ** (1 / 2),
               # r2_score(Y_test, MARS_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, MARS_predictions))])

# Random forest
RF = RandomForestRegressor(n_estimators=1309, min_samples_split=3, min_samples_leaf=9, max_features='sqrt',
                           max_depth=10, bootstrap=True, random_state=seed)
RF.fit(X_train, Y_train)
RF_predictions = RF.predict(X_test)
result.append(['RF', mean_absolute_error(Y_test, RF_predictions),
               mean_absolute_percentage_error(Y_test, RF_predictions),
               mean_squared_error(Y_test, RF_predictions) ** (1 / 2),
               # r2_score(Y_test, RF_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, RF_predictions))])

# Multilayer perceptron network
MLP = MLPRegressor(activation='relu', solver='adam')
MLP.fit(X_train, Y_train)
MLP_predictions = MLP.predict(X_test)
result.append(['MLP', mean_absolute_error(Y_test, MLP_predictions),
               mean_absolute_percentage_error(Y_test, MLP_predictions),
               mean_squared_error(Y_test, MLP_predictions) ** (1 / 2),
               # r2_score(Y_test, MLP_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, MLP_predictions))])

# Gradient boosting regression
GBR = GradientBoostingRegressor(n_estimators=400, min_samples_split=4, min_samples_leaf=5, max_features='log2',
                                max_depth=10, loss='ls', criterion='mae')
GBR.fit(X_train, Y_train)
GBR_predictions = GBR.predict(X_test)
result.append(['GBR', mean_absolute_error(Y_test, GBR_predictions),
               mean_absolute_percentage_error(Y_test, GBR_predictions),
               mean_squared_error(Y_test, GBR_predictions) ** (1 / 2),
               # r2_score(Y_test, GBR_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, GBR_predictions))])

print('MAE'.rjust(9), 'MAPE'.rjust(9), 'RMSE'.rjust(8), 'Improvement'.rjust(18))
print(tabulate(result))

exit()

# Including under/overestimation
# ----  -------  -------  -------  --------  --------
# LR    53.8461  84.6693  71.0762  0.229634   631.606
# MARS  53.8461  84.6693  71.0762  0.229634   631.606
# RF    39.8645  67.5816  56.9081  0.506148  1271.74
# MLP   58.3685  86.8881  76.7941  0.1007     333.122
# GBR   39.872   67.6068  56.9082  0.506146  1271.74
# ----  -------  -------  -------  --------  --------

# Excluding over/underestimation
#       MAE      MAPE     RMSE     Improvement
# ----  -------  -------  -------  ---------
# LR    54.1755  81.555   66.9023  -32.2897
# MARS  54.1755  81.555   66.9023  -32.2897
# RF    37.7966  65.5361  50.5671    7.70551
# MLP   56.6277  83.9729  71.4712  -38.2776
# GBR   37.8395  64.9577  50.6753    7.60076
# ----  -------  -------  -------  ---------

# Determine statistics per operation type
test_individual = pd.concat([X_test['Operatietype'], Y_test, baseline], axis=1)

order = ['AVR', 'AVR + MVP shaving', 'CABG', 'CABG + AVR', 'CABG + pacemaker tijdelijk',
         'Lobectomie of segmentresectie', 'Mediastinoscopie', 'MVB', 'Rethoractomie', 'Wondtoilet']
i = 0
results_per_operation_LR = []
results_per_operation_MARS = []
results_per_operation_RF = []
results_per_operation_MLP = []
results_per_operation_GBR = []
nobservations = []
groups = test_individual.groupby(['Operatietype'])
keys = groups.groups.keys()
for key in keys:
    group = groups.get_group(key)
    X = pd.DataFrame(group['Operatietype'])
    Y = pd.DataFrame(group['Operatieduur'])
    Y_list = Y.values
    initial = pd.DataFrame(group['Geplande operatieduur'])
    nobservations.append([order[i], len(group)])  # amount of observations

    initial_error = mean_absolute_error(Y, initial)
    print('Initial error of', order[i], ':', initial_error)

    pred_LR  = LR.predict(X)
    mae_LR   = mean_absolute_error(Y, pred_LR)
    mape_LR  = mean_absolute_percentage_error(Y, pred_LR)
    # r2_LR    = r2_score(Y, pred_LR)
    imp_LR   = calc_improvement(initial_error, mae_LR)
    error_LR = [abs(Y_list[i][0] - pred_LR[i]) for i in range(0,len(pred_LR))]
    std_LR   = stats.stdev(error_LR)
    results_per_operation_LR.append([order[i], mae_LR, mape_LR, imp_LR, std_LR])

    pred_MARS  = MARS.predict(X)
    mae_MARS   = mean_absolute_error(Y, pred_MARS)
    mape_MARS  = mean_absolute_percentage_error(Y, pred_MARS)
    # r2_MARS    = r2_score(Y, pred_MARS)
    imp_MARS   = calc_improvement(initial_error, mae_MARS)
    error_MARS = [abs(Y_list[i][0] - pred_MARS[i]) for i in range(0,len(pred_MARS))]
    std_MARS   = stats.stdev(error_MARS)
    results_per_operation_MARS.append([order[i], mae_MARS, mape_MARS, imp_MARS, std_MARS])

    pred_RF  = RF.predict(X)
    mae_RF   = mean_absolute_error(Y, pred_RF)
    mape_RF  = mean_absolute_percentage_error(Y, pred_RF)
    # r2_RF    = r2_score(Y, pred_RF)
    imp_RF   = calc_improvement(initial_error, mae_RF)
    error_RF = [abs(Y_list[i][0] - pred_RF[i]) for i in range(0,len(pred_RF))]
    std_RF   = stats.stdev(error_RF)
    results_per_operation_RF.append([order[i], mae_RF, mape_RF, imp_RF, std_RF])

    pred_MLP  = MLP.predict(X)
    mae_MLP   = mean_absolute_error(Y, pred_MLP)
    mape_MLP  = mean_absolute_percentage_error(Y, pred_MLP)
    # r2_MLP    = r2_score(Y, pred_MLP)
    imp_MLP   = calc_improvement(initial_error, mae_MLP)
    error_MLP = [abs(Y_list[i][0] - pred_MLP[i]) for i in range(0,len(pred_MLP))]
    std_MLP   = stats.stdev(error_MLP)
    results_per_operation_MLP.append([order[i], mae_MLP, mape_MLP, imp_MLP, std_MLP])

    pred_GBR  = GBR.predict(X)
    mae_GBR   = mean_absolute_error(Y, pred_GBR)
    mape_GBR  = mean_absolute_percentage_error(Y, pred_GBR)
    # r2_GBR    = r2_score(Y, pred_GBR)
    imp_GBR   = calc_improvement(initial_error, mae_GBR)
    error_GBR = [abs(Y_list[i][0] - pred_GBR[i]) for i in range(0,len(pred_GBR))]
    std_GBR   = stats.stdev(error_GBR)
    results_per_operation_GBR.append([order[i], mae_GBR, mape_GBR, imp_GBR, std_GBR])

    i += 1

print("Amount of observations per surgery in the test set")
print(tabulate(nobservations))
# -----------------------------  ---
# AVR                             78
# AVR + MVP shaving               12
# CABG                           237
# CABG + AVR                      33
# CABG + pacemaker tijdelijk      73
# Lobectomie of segmentresectie    8
# Mediastinoscopie                13
# MVB                             14
# Rethoractomie                    9
# Wondtoilet                      25
# -----------------------------  ---
print()
print("LR")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(9), "Imp".rjust(8), "Std".rjust(11))
print(tabulate(results_per_operation_LR))

print()
print("MARS")
print("Surgery type", "MAE".rjust(21), "MAPE".rjust(9), "Imp".rjust(8), "Std".rjust(11))
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
# Surgery type                   MAE      MAPE      Imp          Std
# -----------------------------  -------  --------  ----------  -------
# AVR                            50.8531   26.9588   -36.9189   30.8287
# AVR + MVP shaving              31.2355   14.4447   -24.1145   16.152
# CABG                           42.2516  114.642      1.45045  35.9865
# CABG + AVR                     73.8985   24.1328   -13.5841   52.4732
# CABG + pacemaker tijdelijk     74.2296   29.6428   -72.682    39.4199
# Lobectomie of segmentresectie  31.875    18.6091    12.069    20.1403
# Mediastinoscopie               85.0915   30.5295   -58.7073   58.9864
# MVB                            97.4652  191.119   -409.146    11.3324
# Rethoractomie                  56.1587   87.753   -156.562    24.3059
# Wondtoilet                     70.1032  180.917   -280.169    19.1505
# -----------------------------  -------  --------  ----------  -------
#
# MARS
# Surgery type                   MAE      MAPE      Imp         Std
# -----------------------------  -------  --------  ----------  -------
# AVR                            50.8531   26.9588   -36.9189   30.8287
# AVR + MVP shaving              31.2355   14.4447   -24.1145   16.152
# CABG                           42.2516  114.642      1.45045  35.9865
# CABG + AVR                     73.8985   24.1328   -13.5841   52.4732
# CABG + pacemaker tijdelijk     74.2296   29.6428   -72.682    39.4199
# Lobectomie of segmentresectie  31.875    18.6091    12.069    20.1403
# Mediastinoscopie               85.0915   30.5295   -58.7073   58.9864
# MVB                            97.4652  191.119   -409.146    11.3324
# Rethoractomie                  56.1587   87.753   -156.562    24.3059
# Wondtoilet                     70.1032  180.917   -280.169    19.1505
# -----------------------------  -------  --------  ----------  -------
#
# RF
# Surgery type                   MAE      MAPE      Imp       Std
# -----------------------------  -------  --------  --------  --------
# AVR                            34.141    15.9264   8.07732  25.2135
# AVR + MVP shaving              24.728    10.2643   1.74302  22.6567
# CABG                           42.0027  115.703    2.0308   35.93
# CABG + AVR                     52.9545   19.7489  18.6075   34.0664
# CABG + pacemaker tijdelijk     36.6324   18.6891  14.7812   37.8587
# Lobectomie of segmentresectie  31.875    18.7729  12.069    20.2897
# Mediastinoscopie               47.2683   17.296   11.8381   40.2725
# MVB                            14.1557   31.0323  26.0522    9.93873
# Rethoractomie                  18.483    27.9072  15.5599   14.408
# Wondtoilet                     16.1534   41.4185  12.4003    9.76178
# -----------------------------  -------  --------  --------  --------
#
# MLP
# Surgery type                   MAE      MAPE       Imp         Std
# -----------------------------  --------  --------  ----------  -------
# AVR                             44.127    18.5735   -18.8093   34.4486
# AVR + MVP shaving               34.1442   13.7192   -35.6724   29.83
# CABG                            46.3146  107.453     -8.02637  38.6027
# CABG + AVR                      79.8412   25.9384   -22.7182   55.4327
# CABG + pacemaker tijdelijk      73.8351   29.5118   -71.7644   39.4014
# Lobectomie of segmentresectie   31.875    19.5916    12.069    23.0269
# Mediastinoscopie                67.2086   23.2298   -25.3531   57.8388
# MVB                            125.452   244.733   -555.345    11.3324
# Rethoractomie                   93.3218  138.79    -326.343    24.3059
# Wondtoilet                     116.443   288.09    -531.469    19.1505
# -----------------------------  --------  --------  ----------  -------
#
# GBR
# Surgery type                   MAE      MAPE      Imp        Std
# -----------------------------  -------  --------  ---------  --------
# AVR                            34.2035   15.8364   7.9092    25.5707
# AVR + MVP shaving              24.9638   10.3063   0.806122  23.2019
# CABG                           42.0422  115.444    1.93879   35.9525
# CABG + AVR                     53.0303   19.5926  18.4909    34.8824
# CABG + pacemaker tijdelijk     37.0959   18.6459  13.703     37.9672
# Lobectomie of segmentresectie  31.875    18.1025  12.069     20.5977
# Mediastinoscopie               46.9231   17.4838  12.4821    38.1739
# MVB                            11.2143   24.1175  41.4179     7.31888
# Rethoractomie                  20        28.2176   8.62944   13.2476
# Wondtoilet                     16.16     36.7293  12.3644    11.1642
# -----------------------------  -------  --------  ---------  --------