#!/usr/bin/env python
# coding: utf-8

# ## Population Growth

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# https://data.worldbank.org/indicator/SP.POP.TOTL


# In[2]:


def read_data(filename):
    """
    Function to read World Bank data from a CSV file and perform data preprocessing.

    - Input: The path to the CSV file.

    - df: A transposed DataFrame with Country Name as Header.
    - countries_df: A DataFrame with 'Country Name' as the index.

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


# In[3]:


filename = 'API_SP.POP.TOTL_DS2_en_csv_v2_6298256.csv'
df, countries_df = read_data(filename)


# In[4]:


df.head()


# In[5]:


countries_df.head()


# In[6]:


data = countries_df[['Country Name', 'Indicator Name'] + list(map(str, range(2000, 2021)))]
data = data.dropna()


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


growth = data[["Country Name", "2020"]].copy()


# In[10]:


growth["Percentage Growth"] = 100.0 * (data["2020"] - data["2000"]) / data["2000"]


# In[11]:


growth['2020'] = pd.to_numeric(growth['2020'], errors='coerce')
growth['Percentage Growth'] = pd.to_numeric(growth['Percentage Growth'], errors='coerce')

print(growth.dtypes)


# In[12]:


growth.describe()


# In[13]:


plt.figure(figsize=(8, 8))
plt.scatter(growth["2020"], growth["Percentage Growth"], 10, marker="o")
plt.xlabel("Population Growth in 2020")
plt.ylabel("Percentage Growth from 2000 to 2021")
plt.title("Scatter Plot of Population Growth in 2020 vs. Percentage Growth (2000 to 2021)")
plt.show()


# In[14]:


import sklearn.preprocessing as pp
scaler = pp.RobustScaler()

df_ex = growth[["2020", "Percentage Growth"]]
scaler.fit(df_ex)
df_norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 1])
plt.xlabel("Normalized Population Growth in 2020")
plt.ylabel("Percentage Growth from 2000 to 2021")
plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import warnings
warnings.filterwarnings("ignore")

def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score

for ic in range(2, 11):
    score = one_silhouette(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")


# In[16]:


kmeans = KMeans(n_clusters=3, n_init=20)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(growth["2020"], growth["Percentage Growth"], 10, labels, marker="o")
plt.scatter(xkmeans, ykmeans, 50, "k", marker="d")
plt.xlabel("Population Growth in 2020")
plt.ylabel("Percentage Growth from 2000 to 2021")
plt.show()


# ### Prediction of Population Growth in India

# In[17]:


india_data = data[data['Country Name'] == 'India']


# In[18]:


india_data


# In[19]:


india_data = india_data.drop(columns=['Country Name', 'Indicator Name']).transpose()
india_data.columns = ["Population Growth"]


# In[20]:


india_data.reset_index(inplace=True)
india_data.rename(columns={'index': 'Year'}, inplace=True)
india_data["Year"] = pd.to_numeric(india_data["Year"], errors='coerce')


# In[21]:


india_data.plot("Year", "Population Growth")
plt.show()


# In[22]:


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    # makes it easier to get a guess for initial parameters
    t = t - 2010
    f = n0 * np.exp(g*t)
    return f


# In[23]:


import scipy.optimize as opt

param, covar = opt.curve_fit(exponential, india_data["Year"], india_data["Population Growth"], p0=(1.2e12, 0.1))
print(f"Population Growth 2010: {param[0]/1e6:6.1f} million")
print(f"growth rate: {param[1]*100:4.2f}%")


# In[24]:


india_data["fit"] = exponential(india_data["Year"], *param)
india_data.plot("Year", ["Population Growth", "fit"])
plt.show()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt

def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for number or array. It uses error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically.
    """
    
    # initiate sigma the same shape as parameter
    var = np.zeros_like(x)   # initialise variance vector
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respect to the jth parameter
            deriv2 = deriv(x, func, parameter, j)
            # multiplied with the i-jth covariance
            # variance vector 
            var = var + deriv1 * deriv2 * covar[i, j]

    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta. Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.
    """

    # create vector with zeros and insert delta value for the relevant parameter
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    
    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """
    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar / matrix
    
    return corr


years_future = np.arange(2021, 2041, 1)
predictions = exponential(years_future, *param)
confidence_range = error_prop(years_future, exponential, param, covar)

plt.figure(figsize=(10, 6))
plt.plot(india_data["Year"], india_data["Population Growth"], 'o', label="Data")
plt.plot(years_future, predictions, label="Best Fitting Function", color='blue')
plt.fill_between(years_future, predictions - confidence_range, predictions + confidence_range, color='blue', alpha=0.2, label="Confidence Range")
plt.xlabel("Year")
plt.ylabel("Population Growth")
plt.title("Exponential Growth Model and Confidence Range")
plt.legend()
plt.show()


# In[ ]:




