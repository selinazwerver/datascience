import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.metrics import classification_report, confusion_matrix

try:
    from IPython.display import clear_output

    have_ipython = True
except ImportError:
    have_ipython = False

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = 1e10 * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window), min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
                # print("i=%d j=%d cost=%0.4g" % (i, j, cost[i, j]))

        # print("Cost matrix:")
        # print(self._print_cost_matrix(cost))

        # Return DTW distance given window
        return cost[-1, -1], cost

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if (np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(np.shape(dm)[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            p = ProgressBar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j], _ = self._dtw_distance(x[i, ::self.subsample_step],
                                                     y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    def _print_cost_matrix(self, cost):
        [i, j] = cost.shape
        print("i=%d, j=%d" % (i, j))
        cost[0][0] = 1
        cost[0][1] = 1
        for row in cost:
            r = "  "
            for c in row:
                r += ("%.4g" % c).rjust(11)
            print(r)

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """

        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()


class ProgressBar:
    """This progress bar was taken from PYMC
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print('\r', self, end="", flush=True)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


print("[Exercise 4.9]")
file_countries = "Earth Surface Temperature Study/GlobalLandTemperaturesByCountry.csv"
data_countries = pd.read_csv(file_countries)

countries = ["Norway", "Finland", "Singapore", "Cambodia"]
data_countries = data_countries[data_countries["Country"].isin(countries)]  # remove data from other countries

# Determine which years to plot to avoid NaN
index_nan = data_countries["Country"].isin(countries) & data_countries["AverageTemperature"].isna()
countries_grouped = data_countries[index_nan].groupby("Country")

dates = []
for country, group in countries_grouped:
    dates.append(group["dt"].values[-2])  # last date where nan occurs per country

dates.sort(reverse=True)  # sort newest date first
start_date = dates[0]  # minimum date from where to plot
npoints = 252  # amount of data points to plot

temperature_per_country = {}
plt.figure(figsize=(12, 6))
for country, group in data_countries.groupby("Country"):
    start_index = pd.Series(group["dt"] == start_date)  # find start index
    start_index = start_index[start_index].index.values
    plot_range = [int(start_index) - group.first_valid_index() + 1,
                  int(start_index) - group.first_valid_index() + npoints]

    temperatures = group["AverageTemperature"].to_numpy()[plot_range[0]:plot_range[1]]
    timeline = group["dt"].to_numpy()[plot_range[0]:plot_range[1]]

    # Plot
    plt.plot(timeline, temperatures, label=country)

    # Store data in dict
    temperature_per_country[country] = temperatures

plt.xticks(rotation=90)
xticks = plt.gca().xaxis.get_major_ticks()
# Plot only every 12 labels
for i in range(len(xticks)):
    if i % 12 != 0:
        xticks[i].set_visible(False)
plt.tight_layout()
plt.legend()
plt.title("Yearly temperatures for %i years" % (npoints / 12 - 1))
plt.grid()
plt.savefig("4.9a_temperatures.png", dpi=300)

print("[4.9a] Plot saved")
print()

dtw = KnnDtw()
print("[4.9b] Table with minimal DTW distance")
HeaderRow = "DISTANCE ".ljust(10)
for i1, c1 in enumerate(countries):
    HeaderRow += c1.ljust(10)
print(HeaderRow)

for i1, c1 in enumerate(countries):
    Row = (c1 + " ").rjust(10)
    for i2, c2 in enumerate(countries):
        distance, cost = dtw._dtw_distance(temperature_per_country[c1], temperature_per_country[c2])
        Row += str(int(distance)).ljust(10)
    print(Row)
