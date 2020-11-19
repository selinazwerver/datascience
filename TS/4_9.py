import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("[Exercise 4.9]")
file_countries = "Earth Surface Temperature Study/GlobalLandTemperaturesByCountry.csv"
data_countries = pd.read_csv(file_countries)

countries = ["Norway", "Finland", "Singapore", "Cambodia"]
data_countries = data_countries[data_countries["Country"].isin(countries)] # remove data from other countries

# Determine which years to plot to avoid NaN
index_nan = data_countries["Country"].isin(countries) & data_countries["AverageTemperature"].isna()
countries_grouped = data_countries[index_nan].groupby("Country")

dates = []
for country, group in countries_grouped:
    dates.append(group["dt"].values[-2]) # last date where nan occurs per country

dates.sort(reverse=True) # sort newest date first
start_date = dates[0] # date from where to plot
npoints = 180
# print("Plotting from:", start_date)

plt.figure()
for country, group in data_countries.groupby("Country"):
    start_index = pd.Series(group["dt"] == start_date) # find start index
    start_index = start_index[start_index].index.values
    # plot_range = [int(start_index) - group.first_valid_index(), group.shape[0]]
    plot_range = [int(start_index) - group.first_valid_index(), int(start_index) - group.first_valid_index() + npoints]

    temperatures = group["AverageTemperature"].to_numpy()[plot_range[0]:plot_range[1]]
    timeline = group["dt"].to_numpy()[plot_range[0]:plot_range[1]]
    # date_time = pd.to_datetime(group["dt"])
    print(timeline)

    # plt.plot(np.arange(start=plot_range[0], stop=plot_range[1]), temperatures[plot_range[0]:plot_range[1]])
    plot_set = pd.DataFrame()
    plot_set["temp"] = temperatures
    plot_set["date"] = timeline

    plt.xticks(rotation=90)
    plt.plot( plot_set["date"], plot_set["temp"])

    # plt.plot(timeline[plot_range[0]:plot_range[1]], temperatures[plot_range[0]:plot_range[1]])
    # plt.gca().axes.xaxis.set_visible(False)
    plt.show()
    exit()
    # temperatures = temperatures[start_index:]



