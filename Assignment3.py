# coding: utf-8


# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import errors as err
import sklearn.preprocessing as pp
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import scipy.optimize as opt
import warnings
warnings.filterwarnings("ignore")


def read_data(filename):
    """
    Function to read World Bank data from a CSV file and perform data preprocessing.

    - Input:
        filename: The path to the CSV file.

    - Returns :
        df: A transposed DataFrame with Country Name as Header.
        countries_df: A DataFrame with 'Country Name' as the index.

    """
    df = pd.read_csv(filename, skiprows=4)
    df = df.iloc[:, :-1]
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    countries_df = df.T
    countries_df.reset_index(inplace=True)
    countries_df.rename(columns={'index': 'Country Name'}, inplace=True)
    return df, countries_df


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


def exponential(t, n0, g):
    """Calculates exponential function with
       scale factor n0 and growth rate g."""
    # makes it easier to get a guess for initial parameters
    t = t - 2010
    f = n0 * np.exp(g*t)
    return f


filename = 'API_SP.POP.TOTL_DS2_en_csv_v2_6298256.csv'
df, countries_df = read_data(filename)
print(df.head())
print(countries_df.head())

data = \
    countries_df[
        ['Country Name', 'Indicator Name'] + \
            list(map(str, range(2000, 2021)))]
data = data.dropna()
print(data.head())

growth = data[["Country Name", "2020"]].copy()
growth["Percentage Growth"] = \
    100.0 * (data["2020"] - data["2000"]) / data["2000"]
growth['2020'] = pd.to_numeric(growth['2020'], errors='coerce')
growth['Percentage Growth'] = \
    pd.to_numeric(growth['Percentage Growth'], errors='coerce')
print(growth.dtypes)
print(growth.describe())

plt.figure(figsize=(8, 8))
plt.scatter(growth["2020"], growth["Percentage Growth"], 10, marker="o")
plt.xlabel("Population Growth in 2020")
plt.ylabel("Percentage Growth from 2000 to 2021")
plt.title("""Scatter Plot of Population Growth in 2020 vs.
    Percentage Growth (2000 to 2021)""")
plt.show()

scaler = pp.RobustScaler()
df_ex = growth[["2020", "Percentage Growth"]]
scaler.fit(df_ex)
df_norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 1])
plt.xlabel("Normalized Population Growth in 2020")
plt.ylabel("Percentage Growth from 2000 to 2021")
plt.title("""Scatter Plot of Normalized Population Growth in 2020 vs.
    Normalized Percentage Growth (2000 to 2021)""")
plt.show()

for ic in range(2, 11):
    score = one_silhouette(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

kmeans = KMeans(n_clusters=3, n_init=20)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(
    growth["2020"], growth["Percentage Growth"], 10, c=labels, marker="o")
plt.xlabel("Population Growth in 2020")
plt.ylabel("Growth from 2000 to 2021")
plt.legend()
plt.title("""Scatter Plot of Population Growth in 2020 vs.
    Percentage Growth (2000 to 2021) with clusters""")
plt.show()

india_data = data[data['Country Name'] == 'India']
print(india_data)

india_data = \
    india_data.drop(columns=['Country Name', 'Indicator Name']).transpose()
india_data.columns = ["Population Growth"]
india_data.reset_index(inplace=True)
india_data.rename(columns={'index': 'Year'}, inplace=True)
india_data["Year"] = pd.to_numeric(india_data["Year"], errors='coerce')
india_data.plot("Year", "Population Growth")
plt.title("Population Growth with year")
plt.show()

param, covar = opt.curve_fit(
    exponential, india_data["Year"], india_data["Population Growth"],
    p0=(1.2e12, 0.1))
print(f"Population Growth 2010: {param[0]/1e6:6.1f} million")
print(f"growth rate: {param[1]*100:4.2f}%")

india_data["fit"] = exponential(india_data["Year"], *param)
india_data.plot("Year", ["Population Growth", "fit"])
plt.title("Population Growth with year(fit)")
plt.show()

years = np.arange(2021, 2040, 1)
predictions = exponential(years, *param)
confidence_range = err.error_prop(years, exponential, param, covar)

plt.figure(figsize=(10, 6))
plt.plot(india_data["Year"],
         india_data["Population Growth"], 'o', label="Data")
plt.plot(
    years, predictions, label="Best Fitting Function", color='blue')
plt.fill_between(
    years,
    predictions - confidence_range,
    predictions + confidence_range,
    color='blue', alpha=0.2, label="Confidence Range")
plt.xlabel("Year")
plt.ylabel("Population Growth")
plt.title("Exponential Growth Model and Confidence Range")
plt.legend()
plt.show()
