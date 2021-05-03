import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

print('[Analysis]')

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

plt.figure()
for name, values in data.iteritems():
    nan_count.append([name, (values.isna().sum() / len(values)) * 100])
    groups = data.fillna(-1).groupby(name)
    keys = groups.groups.keys()
    # print(name)
    # print('Fraction of NaN: %.2f' % (values.isna().sum() / len(values)))
    # print('Amount of groups:', len(keys))
    # print()
    # if name == 'Benadering':
    #     for key in keys - [-1]:
    #         group = groups.get_group(key)
    #         if len(group['Operatieduur']) > 1:
    #             group['Operatieduur'].plot.kde()
        # plt.title('kde for', name)
        # plt.savefig('figures/variation_benadering.png', dpi=300)

# plt.figure(figsize=(20, 10))
# plt.barh(categories, nan_count)
# plt.grid()
# plt.xlabel('Percentage of NaN in category [%]')
# plt.savefig("figures/nan_percentage.png", dpi=300)

# Determine largest deviation in the data set
data['Difference'] = abs(data['Operatieduur'] - data['Geplande operatieduur'])  # generate difference column
procentual_diff = ((data['Difference']/data['Geplande operatieduur']) * 100).to_list()
print('max:', max(procentual_diff), 'index:', procentual_diff.index(max(procentual_diff)), 'planned:',
      data['Geplande operatieduur'].loc[procentual_diff.index(max(procentual_diff))], 'real:',
      data['Operatieduur'].loc[procentual_diff.index(max(procentual_diff))])
print((sum(i > 100 for i in procentual_diff)/len(procentual_diff))*100)

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

    # if show: # print results
    #     for feature, nkeys, var, frac in results:
    #         print(feature, '& %.2f' % frac)

    return results

## Percentage over/underestimating per surgery
data['Difference'] = (data['Operatieduur'] - data['Geplande operatieduur'])  # generate difference column
data['Percentual diff'] = (data['Difference']/data['Geplande operatieduur']) * 100

groups = data.groupby('Operatietype')
keys = ['AVR', 'AVR + MVP shaving', 'CABG', 'CABG + AVR', 'CABG + Pacemakerdraad tijdelijk',
         'Lobectomie of segmentresectie', 'Mediastinoscopie', 'MVP', 'Rethoracotomie', 'Wondtoilet']
average_percentage = []

for key in keys:
    group = groups.get_group(key)
    average_percentage.append([key, np.mean(group['Percentual diff'])])

print(tabulate(average_percentage))
exit()

# Check which columns are numerical/categorical
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
categorial_variance = calc_variance_categorial(data, categorical_cols, target, True)
numerical_corr = data[numerical_cols].corr()
print('Feature', 'Total'.rjust(29), 'Weighted'.rjust(11), 'NaN'.rjust(4), 'With NaN'.rjust(16))
print(tabulate(categorial_variance))

# plt.figure(figsize=(10,7))
# sn.heatmap(numerical_corr, annot=True)
# plt.tight_layout()
# plt.savefig("figures/correlation_matrix.png", dpi=300)

print()

operation_groups = data.fillna(-1).groupby('Operatietype')
noperations = len(data['Operatietype'])
max_group_size = 0
for key in operation_groups.groups.keys() - [-1]:
    group_size = len(operation_groups.get_group(key))
    if group_size > max_group_size: max_group_size = group_size


foperations_per_threshold = []
ntypes_per_threshold = []
nthresholds = 300
for threshold in range(1,nthresholds):
    total_operations = 0
    operation_types = 0
    for key in operation_groups.groups.keys() - [-1]:
        if len(operation_groups.get_group(key)) < threshold:
            continue
        else:
            total_operations += len(operation_groups.get_group(key))
            operation_types += 1
    foperations_per_threshold.append(total_operations/noperations)
    ntypes_per_threshold.append(operation_types)

fig,ax = plt.subplots()
ax.plot(range(1,nthresholds), foperations_per_threshold, color='coral', label='fraction')
ax.set_xlabel('Threshold')
ax.set_ylabel('Fraction of total operations')
ax2 = ax.twinx()
ax2.plot(range(1,nthresholds), ntypes_per_threshold, color='darkmagenta', label='types')
ax2.hlines(y=10, xmin=-10, xmax=nthresholds+10, colors='k')
ax2.set_ylabel('Amount of operation types')
fig.legend(bbox_to_anchor=(0.7, 0.85), loc='upper left', ncol=1)
ax2.set_xlim(right=310, left=-10)
plt.savefig('figures/threshold_operations.png', dpi=300)

noperations_goal = 10  # amount of operations to predict
threshold = [n for n,i in enumerate(ntypes_per_threshold) if i < noperations_goal][0]
print('Threshold for', noperations_goal, 'is:', threshold)
print('Percentage of data left:', foperations_per_threshold[threshold])

target = 'Operatieduur'
groups = data.fillna(-1).groupby('Operatietype')
keys = groups.groups.keys() - [-1]
stats = []
for key in keys:
    group = groups.get_group(key)
    if len(group) < threshold:
        continue
    stats.append([np.mean(group[target]), np.std(group[target])])

stats = sorted(stats, key=lambda x: x[0])
mean = [l[0] for l in stats]
stdev = [l[1] for l in stats]

fit, result = np.polynomial.polynomial.polyfit(mean, stdev, deg=1, full=True)  # fit line
fitted_stdev = [(fit[0] + value*fit[1]) for value in mean]  # calculate stdev according to fit
corr = np.corrcoef(mean, stdev)[0,1]  # calculate correlation from correlation matrix
plt.clf()
plt.plot(mean, stdev, 'o')
plt.plot(mean, fitted_stdev)
plt.grid()
plt.xlabel('Mean')
plt.ylabel('Standard deviation')
plt.tight_layout()
plt.savefig('figures/stdev_mean_operatietype.png', dpi=300)

print('Correlation:', corr)



