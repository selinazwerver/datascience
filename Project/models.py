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
nfeatures = 4  # best result : 1
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
MLP.fit(X_train, Y_train)
MLP_predictions = MLP.predict(X_test)
result.append(['MLP', mean_absolute_error(Y_test, MLP_predictions),
               mean_absolute_percentage_error(Y_test, MLP_predictions),
               mean_squared_error(Y_test, MLP_predictions) ** (1 / 2),
               # r2_score(Y_test, MLP_predictions),
               calc_improvement(mean_absolute_error(Y_test, baseline), mean_absolute_error(Y_test, MLP_predictions))])

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
print(keys)

for key in keys:
    group = groups.get_group(key)
    X = pd.DataFrame(group[features[0:nfeatures]])
    Y = pd.DataFrame(group['Operatieduur'])
    Y_list = Y.values
    initial = pd.DataFrame(group['Geplande operatieduur'])
    # print(initial)
    # print(np.mean(initial.values))
    # exit()
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

    mean_duration.append([order[i], np.mean(initial.values), stats.stdev(initial.values.flatten()),
                          np.mean(pred_LR), stats.stdev(pred_LR), np.mean(pred_MARS), stats.stdev(pred_MARS),
                          np.mean(pred_RF), stats.stdev(pred_RF), np.mean(pred_MLP), stats.stdev(pred_MLP),
                          np.mean(pred_GBR), stats.stdev(pred_GBR)])

    i += 1

print()
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
print("Mean duration and standard deviation")
# print("Surgery type", "Initial mean".rjust(21), )
print(tabulate(mean_duration))

#                                Mean in  Stdev in  Mean LR  Stdev LR  MeanMARS StdevMARS Mean RF   Stdev RF  MeanMLP  StdevMLP  MeanGBR  StdevGBR
# -----------------------------  -------  --------  -------  --------  -------  --------  --------  --------  -------  --------  -------  --------
# AVR                            211.147  38.4218   221.764  15.3431   218.526  14.815    213.081    6.04332  207.543  12.3995   213.069   6.66047
# AVR + MVP shaving              225.143  28.7639   223.125  10.5646   221.773  10.203    213.205    7.37971  222.37    1.66794  214.266   6.23398
# CABG                           226.713  25.5649   226.309  18.2439   226.739  17.6003   227.166   22.0159   228.88   28.2591   227.847  23.2988
# CABG + AVR                     246.528  49.7028   264.591   9.18305  265.688   8.86874  266.885    5.23939  241.656   1.30258  266.368   2.71198
# CABG + pacemaker tijdelijk     241.725  40.8566   257.662  18.5566   260.93   17.9215   261.443   10.8288   255.2     2.63222  262.211  11.8553
# Lobectomie of segmentresectie  156      12.7279   196.742  37.8536   204.081  36.5047   185.761    8.6561   182.746  89.199    189.599  10.4221
# Mediastinoscopie               222.95   53.2388   232.34   16.4771   240.35   15.89     234.962   18.4711   270.527  38.8705   236.383  19.3768
# MVB                             55       7.07107  230.466   0        142.091   0         52.9129   0        269.963   0         57.343   0
# -----------------------------  -------  --------  -------  --------  -------  --------  --------  --------  -------  --------  -------  --------

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
# -----------------------------  --------  --------  ----------  -------
# AVR                             38.4632   24.3623   -24.1816   32.1299
# AVR + MVP shaving               29.3478   13.2542   -16.3934   20.9724
# CABG                            34.6957   24.4646     4.69144  30.2892
# CABG + AVR                      49.4656   17.3963    28.3973   44.0589
# CABG + pacemaker tijdelijk      34.4615   17.3627    16.4203   30.5547
# Lobectomie of segmentresectie   50.758    20.312     44.5268    1.0373
# Mediastinoscopie                57.5727   40.7425   -26.2559   48.8231
# MVB                            169.966   320.228   -623.258    26.163
# -----------------------------  --------  --------  ----------  -------
#
# MARS
# Surgery type                   MAE      MAPE      Imp        Std
# -----------------------------  -------  --------  ---------  --------
# AVR                            37.4516   23.5902   -20.9158  31.5338
# AVR + MVP shaving              29.6237   13.2321   -17.4879  21.4708
# CABG                           34.5894   24.4113     4.9835  30.1728
# CABG + AVR                     49.1634   17.3943    28.8346  43.9257
# CABG + pacemaker tijdelijk     33.9956   17.3724    17.5503  30.5255
# Lobectomie of segmentresectie  43.4187   18.7608    52.5479   2.38612
# Mediastinoscopie               58.1638   42.1678   -27.5522  49.7223
# MVB                            81.591   159.087   -247.196   26.163
# -----------------------------  -------  --------  ---------  --------
#
# RF
# Surgery type                   MAE      MAPE     Imp        Std
# -----------------------------  -------  -------  ---------  -------
# AVR                            34.2199  21.786   -10.4819   30.0076
# AVR + MVP shaving              32.1754  13.4644  -27.6077   25.1581
# CABG                           34.2922  24.9422    5.79988  30.3288
# CABG + AVR                     49.6827  17.2075   28.0829   44.2948
# CABG + pacemaker tijdelijk     33.3296  16.4904   19.1655   30.5054
# Lobectomie of segmentresectie  61.7388  24.0068   32.5259   30.2348
# Mediastinoscopie               48.45    41.4977   -6.25     43.3162
# MVB                            18.5     29.5024   21.2766   10.7297
# -----------------------------  -------  -------  ---------  -------
#
# MLP
# Surgery type                   MAE       MAPE      Imp         Std
# -----------------------------  --------  --------  ----------  -------
# AVR                             33.3995   21.481     -7.83323  30.8704
# AVR + MVP shaving               32.2579   13.0828   -27.9349   18.028
# CABG                            37.0348   25.4287    -1.7339   31.3805
# CABG + AVR                      58.1447   19.3488    15.8339   49.475
# CABG + pacemaker tijdelijk      34.5887   15.5487    16.1118   33.1082
# Lobectomie of segmentresectie   64.7539   31.1083    29.2307   50.3081
# Mediastinoscopie                77.8721   53.4148   -70.7723   63.1386
# MVB                            209.463   392.248   -791.334    26.163
# -----------------------------  --------  --------  ----------  -------
#
# GBR
# Surgery type                   MAE      MAPE     Imp        Std
# -----------------------------  -------  -------  ---------  --------
# AVR                            34.3016  21.8209  -10.7457   30.0326
# AVR + MVP shaving              32.0911  13.3618  -27.2734   24.2126
# CABG                           34.3418  25.173     5.66358  30.603
# CABG + AVR                     49.0686  17.1443   28.9719   44.6093
# CABG + pacemaker tijdelijk     33.3293  16.6504   19.1663   30.4773
# Lobectomie of segmentresectie  57.9007  22.4367   36.7205   49.313
# Mediastinoscopie               48.45    41.8085   -6.25     43.6191
# MVB                            18.5     31.9724   21.2766    4.46467
# -----------------------------  -------  -------  ---------  --------

