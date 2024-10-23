#!/usr/bin/env python
# coding: utf-8

# In[1]:


use a dataset with values.apply mean,median and mode imputations to handle the missing data, then, create pair plots using seaborn before and after imputation. how do the relationships between features change after each imputation method


# In[11]:


import pandas as pd
import numpy as np
np.random.seed(42)
data = {
    'Feature1': np.random.randn(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.randint(1, 10, 100).astype(float)
}
data['Feature1'][np.random.choice(range(100), size=15, replace=False)] = np.nan
data['Feature2'][np.random.choice(range(100), size=10, replace=False)] = np.nan
data['Feature3'][np.random.choice(range(100), size=5, replace=False)] = np.nan  # Now this will work
df = pd.DataFrame(data)
print(df)


# In[8]:


from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(strategy='mean')
df_mean = df.copy()
df_mean['Feature1'] = mean_imputer.fit_transform(df_mean[['Feature1']])
df_mean['Feature2'] = mean_imputer.fit_transform(df_mean[['Feature2']])
median_imputer = SimpleImputer(strategy='median')
df_median = df.copy()
df_median['Feature1'] = median_imputer.fit_transform(df_median[['Feature1']])
df_median['Feature2'] = median_imputer.fit_transform(df_median[['Feature2']])
mode_imputer = SimpleImputer(strategy='most_frequent')
df_mode = df.copy()
df_mode['Feature3'] = mode_imputer.fit_transform(df_mode[['Feature3']])


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)
plt.title('Before Imputation')
plt.show()
sns.pairplot(df_mean)
plt.title('After Mean Imputation')
plt.show()
sns.pairplot(df_median)
plt.title('After Median Imputation')
plt.show()
sns.pairplot(df_mode)
plt.title('After Mode Imputation')
plt.show()


# In[10]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
sns.boxplot(y=df['Feature1'])
plt.title('Before Imputation')
plt.subplot(1, 4, 2)
sns.boxplot(y=df_mean['Feature1'])
plt.title('Mean Imputation')
plt.subplot(1, 4, 3)
sns.boxplot(y=df_median['Feature1'])
plt.title('Median Imputation')
plt.subplot(1, 4, 4)
sns.boxplot(y=df_mode['Feature3'])
plt.title('Mode Imputation')
plt.tight_layout()
plt.show()


# In[12]:


#Apply z-score normalization on the dataset and generate scatter plots of at least two features before and after 
#normalisatiion. How did the normalization affect the range and distribution of these features? What would you 
#prefer min max scaling over z core normalisation or vice versa? can you visualise an exapmle where one scaling 
#method produces better results? Load a dataset with numerical variables. Create scatter plots or the line plots to
#compare the data before and after applying min max scaling what changes do you notiice in the value distribution 
#after scaling correlation matrix. Apply min max scaling and z score normalisationseperately> After each tranformation
#plot a hashmap of the new correlatiuon matrix. How do the relationships between the cariables changes(if at all).
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

X = iris_df.drop('species', axis=1)

z_scaler = StandardScaler()
X_z = z_scaler.fit_transform(X)

min_max_scaler = MinMaxScaler()
X_min_max = min_max_scaler.fit_transform(X)

def plot_scatter(data, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.7)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_scatter(iris_df, 'Scatter Plot Before Scaling')

X_z_df = pd.DataFrame(X_z, columns=X.columns)
X_min_max_df = pd.DataFrame(X_min_max, columns=X.columns)

plot_scatter(X_z_df, 'Scatter Plot After Z-Score Normalization')

plot_scatter(X_min_max_df, 'Scatter Plot After Min-Max Scaling')

def plot_correlation_matrix(data, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()

plot_correlation_matrix(iris_df.drop('species', axis=1), 'Correlation Matrix Before Scaling')

plot_correlation_matrix(X_z_df, 'Correlation Matrix After Z-Score Normalization')

plot_correlation_matrix(X_min_max_df, 'Correlation Matrix After Min-Max Scaling')


# In[ ]:




