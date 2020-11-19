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
start_date = dates[0] # minimum date from where to plot
npoints = 252 # amount of data points to plot

plt.figure(figsize=(12,6))
for country, group in data_countries.groupby("Country"):
    start_index = pd.Series(group["dt"] == start_date) # find start index
    start_index = start_index[start_index].index.values
    plot_range = [int(start_index) - group.first_valid_index(), int(start_index) - group.first_valid_index() + npoints]

    temperatures = group["AverageTemperature"].to_numpy()[plot_range[0]:plot_range[1]]
    timeline = group["dt"].to_numpy()[plot_range[0]:plot_range[1]]

    # Plot
    plt.plot(timeline, temperatures, label=country)

plt.xticks(rotation=90)
xticks = plt.gca().xaxis.get_major_ticks()
# Plot only every 12 labels
for i in range(len(xticks)):
    if i % 12 != 0:
        xticks[i].set_visible(False)
plt.tight_layout()
plt.legend()
plt.title("Yearly temperatures for %i years" % (npoints/12-1))
plt.grid()
plt.savefig("4.9a_temperatures.png", dpi=300)

