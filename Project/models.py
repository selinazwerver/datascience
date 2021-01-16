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
MLP = MLPRegressor(activation='relu', solver='adam', random_state=seed)
MLP.fit(X_train_mlp, Y_train_mlp)
MLP_predictions = MLP.predict(X_test_mlp)
result.append(['MLP', mean_absolute_error(Y_test_mlp, MLP_predictions),
               mean_absolute_percentage_error(Y_test_mlp, MLP_predictions),
               mean_squared_error(Y_test_mlp, MLP_predictions) ** (1 / 2),
               # r2_score(Y_test, MLP_predictions),
               calc_improvement(mean_absolute_error(Y_test_mlp, baseline), mean_absolute_error(Y_test_mlp, MLP_predictions))])


# MLP = MLPRegressor(activation='relu', solver='adam', random_state=seed)
# MLP.fit(X_train, Y_train)
# MLP_predictions = MLP.predict(X_test)
# result.append(['MLP', mean_absolute_error(Y_test, MLP_predictions),
#                mean_absolute_percentage_error(Y_test, MLP_predictions),
#                mean_squared_error(Y_test, MLP_predictions) ** (1 / 2),
#                # r2_score(Y_test, MLP_predictions),
#                calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, MLP_predictions))])

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
# ----  -------  -------  -------  --------
# LR    38.0595  23.5231  51.1522   2.80138
# MARS  37.3377  22.6953  49.8531   4.64491
# RF    36.0134  21.5182  48.3988   8.02691
# MLP   40.3652  24.5227  55.4981  -3.08688
# GBR   35.9821  21.6276  48.4924   8.10679
# ----  -------  -------  -------  --------


# Determine statistics per operation type
test_individual = pd.concat([X_test[features[0:nfeatures]], Y_test, baseline], axis=1)
test_individual['New index'] = range(0, len(test_individual))
test_individual = test_individual.set_index('New index')

order = ['AVR', 'AVR + MVP shaving', 'CABG', 'CABG + AVR', 'CABG + pacemaker tijdelijk',
         'Lobectomie of segmentresectie', 'Mediastinoscopie', 'MVB', 'Rethoractomie', 'Wondtoilet']
i = 0
results_per_operation_LR = []
results_per_operation_MARS = []
results_per_operation_RF = []
results_per_operation_MLP = []
results_per_operation_GBR = []
mean_duration = []
nobservations = []

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

    pred_MLP  = MLP.predict(X_mlp)
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

    mean_duration.append([order[i], np.mean(initial.values), stats.stdev(initial.values.flatten()),
                          np.mean(pred_LR), stats.stdev(pred_LR), np.mean(pred_MARS), stats.stdev(pred_MARS),
                          np.mean(pred_RF), stats.stdev(pred_RF), np.mean(pred_MLP), stats.stdev(pred_MLP),
                          np.mean(pred_GBR), stats.stdev(pred_GBR)])

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
print("Mean duration and standard deviation")
# print("Surgery type", "Initial mean".rjust(21), )
print(tabulate(mean_duration))

#                                Mean in  Stdev in  Mean LR  Stdev LR  MeanMARS StdevMARS Mean RF   Stdev RF  MeanMLP  StdevMLP  MeanGBR  StdevGBR
# -----------------------------  -------  --------  -------  -------  -------  --------  --------  --------  -------  --------  --------  --------
# AVR                            211.147  38.4218   221.696  15.2907  218.482  14.7933   212.302    6.67019  205.857  14.2966   213.222    7.03028
# AVR + MVP shaving              225.143  28.7639   223.032  10.5272  221.703  10.2189   213.166    7.77835  221.57    2.12883  214.64     7.15527
# CABG                           226.713  25.5649   226.183  18.1872  226.758  17.4533   227.215   22.841    227.932  31.1989   227.718   23.2284
# CABG + AVR                     246.528  49.7028   264.311   9.1505  265.659   8.88251  266.207    5.06059  240.913   1.50812  267.379    2.49655
# CABG + pacemaker tijdelijk     241.725  40.8566   257.386  18.4908  260.88   17.9493   260.862    9.78178  255.075   3.04753  262.359   11.8633
# Lobectomie of segmentresectie  156      12.7279   196.627  37.7547  204.736  35.7473   187.053    5.29536  175.27   95.8353   177.119    5.7474
# Mediastinoscopie               222.95   53.2388   232.108  16.4341  240.357  15.5603   236.227   17.0393   270.682  42.2325   235.581   19.5909
# MVB                             55       7.07107  230.215   0       142.157   0         62.2191   0        268.02    0         58.3342   0
# -----------------------------  -------  --------  -------  -------  -------  --------  --------  --------  -------  --------  --------  --------

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
# Surgery type                   MAE       MAPE      Imp         Std
# -----------------------------  --------  --------  ---------  --------
# AVR                             38.4275   24.3403   -24.0662  32.1103
# AVR + MVP shaving               29.3695   13.2519   -16.4797  20.997
# CABG                            34.7114   24.4521     4.6483  30.2847
# CABG + AVR                      49.5217   17.3918    28.3159  44.1142
# CABG + pacemaker tijdelijk      34.487    17.3438    16.3583  30.5608
# Lobectomie of segmentresectie   50.8732   20.3171    44.4008   1.13622
# Mediastinoscopie                57.5341   40.6968   -26.1713  48.7947
# MVB                            169.715   319.771   -622.191   26.163
# -----------------------------  --------  --------  ---------  --------
#
# MARS
# Surgery type                   MAE      MAPE      Imp        Std
# -----------------------------  -------  --------  ----------  --------
# AVR                            37.4449   23.5777   -20.894    31.52
# AVR + MVP shaving              29.6301   13.2324   -17.5132   21.5163
# CABG                           34.6569   24.3986     4.79815  30.2584
# CABG + AVR                     49.1723   17.3945    28.8217   43.928
# CABG + pacemaker tijdelijk     34.0056   17.3747    17.5261   30.5239
# Lobectomie of segmentresectie  42.7639   18.5201    53.2635    3.14355
# Mediastinoscopie               58.0467   42.1469   -27.2953   49.608
# MVB                            81.6569  159.207   -247.476    26.163
# -----------------------------  -------  --------  ----------  --------
#
# RF
# Surgery type                   MAE      MAPE      Imp         Std
# -----------------------------  -------  -------  ---------  --------
# AVR                            34.2482  21.693   -10.5732   29.9559
# AVR + MVP shaving              32.1106  13.4795  -27.3506   25.3002
# CABG                           34.2837  25.04      5.82334  30.4274
# CABG + AVR                     49.7105  17.1828   28.0427   44.4776
# CABG + pacemaker tijdelijk     33.3405  16.3474   19.139    30.5624
# Lobectomie of segmentresectie  60.4466  23.4782   33.9381   33.5955
# Mediastinoscopie               48.45    41.5716   -6.25     43.8603
# MVB                            18.5     34.6912   21.2766    2.43118
# -----------------------------  -------  -------  ---------  --------
#
# MLP
# Surgery type                   MAE      MAPE      Imp         Std
# -----------------------------  --------  --------  ----------  -------
# AVR                             33.5896   21.4365    -8.44683  31.2324
# AVR + MVP shaving               32.4777   13.0958   -28.8067   18.5176
# CABG                            37.6662   25.6327    -3.46843  32.1828
# CABG + AVR                      58.5795   19.4773    15.2046   49.626
# CABG + pacemaker tijdelijk      34.709    15.5582    15.8201   33.1873
# Lobectomie of segmentresectie   72.2299   33.534     21.0602   56.9444
# Mediastinoscopie                79.7126   54.1363   -74.8083   64.2461
# MVB                            207.52    388.705   -783.065    26.163
# -----------------------------  --------  --------  ----------  -------
#
# GBR
# Surgery type                   MAE      MAPE      Imp         Std
# -----------------------------  -------  -------  ---------  --------
# AVR                            34.4062  21.871   -11.0832   30.0615
# AVR + MVP shaving              31.8084  13.3678  -26.1523   24.2442
# CABG                           34.3319  25.152     5.69086  30.5673
# CABG + AVR                     48.899   17.1676   29.2173   44.3774
# CABG + pacemaker tijdelijk     33.3231  16.6623   19.1812   30.4802
# Lobectomie of segmentresectie  70.3814  27.5424   23.0805   33.1435
# Mediastinoscopie               48.45    41.6988   -6.25     43.3673
# MVB                            18.5     32.5251   21.2766    3.06293
# -----------------------------  -------  -------  ---------  --------



