import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import *
import sys

# extract estimated data from results (the data obtained from the graph)
est_data = pd.read_excel("../results/FigureData2.xlsx", sheet_name="Y-values", header=0, na_values=["N/A", "NaN"])
est_data = est_data.dropna()
est_data = est_data[est_data.columns[-1]].astype(float)
est_data = np.array(est_data[:-1])

# extract actual data from solar.csv
actual_data = pd.read_csv("../data/solar.csv") # read
actual_data = actual_data.rename(columns={actual_data.columns[-1]: "Values"}) # rename column
# print(actual_data.columns)
actual_data["Values"] = actual_data['Values'].astype(float) # convert to floats

# filter data

# the N/A value, according to the csv
actual_data = actual_data[actual_data['Values'] != -9999.000] 

# filter only start of year and middle
actual_data["Date"] = pd.to_datetime(actual_data["Date"])
actual_data = actual_data[ ((actual_data["Date"].dt.month == 1) & (actual_data["Date"].dt.day == 1))
                       |   ((actual_data["Date"].dt.month == 7) & (actual_data["Date"].dt.day == 1))]

# filter years between 1960 and 2018
actual_data = actual_data[ (actual_data["Date"].dt.year >= 1960) & (actual_data["Date"].dt.year <= 2018)]

# keep only the y values
actual_data = np.array(actual_data["Values"])

# undetected data, error:
print(f"{actual_data.shape=}")
print(f"{est_data.shape=}")
if actual_data.shape != est_data.shape:
    print("\n\nError: Mismatched vector shapes!")
    print(f"estimated data samples: {est_data.shape[0]}")
    print(f"actual data samples: {actual_data.shape[0]}")
    sys.exit(1)

print("\nestimated data: ")
print(est_data)

print("\nactual data: ")
print(actual_data)


# get the residual - vector of xi-hat - xi for the i-th week
residual = est_data - actual_data

print("\nresidual: ")
print(residual)
print("residual mean: ", np.mean(residual))

# modify the xs so they match the year labels
xs = np.arange(1960, 2019, step=0.5)


# plot the actual data and the estimated on one plot
plt.plot(xs, actual_data, '-b')
plt.plot(xs, est_data, '-r')
plt.title("Plot of actual data and estimated data")
plt.legend(["Actual", "Estimated"])
plt.xlabel("Year")
plt.ylabel("Mean index")
# plt.xticks(range(0, len(xs), 4), xs)
plt.show()


# plot for the residual
plt.subplot(1, 2, 1)
plt.title("Residual plot")
plt.xlabel("Year")
plt.plot(xs, residual, '-or')
# plt.xticks(range(0, len(xs), 4), xs[::4])

# histogram for the residual
plt.subplot(1, 2, 2)
plt.title("Histogram of residual")
plt.hist(residual)
plt.show()

# dtw - estimated vs actual

# define the query and template
query = est_data
template = actual_data

# find the best match with the canonical recursion formula
alignment = dtw(query, template, keep_internals=True)

print(f"\ndistance: {alignment.distance:.2f}") # prints 105.40
print(f"normalized distance: {alignment.normalizedDistance}") # prints 1.10

# display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway", xlab="Estimated data", ylab="Actual data")

# align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)


## see the recursion relation, as formula and diagram
# print(rabinerJuangStepPattern(6,"c"))
# rabinerJuangStepPattern(6,"c").plot()

plt.show()
