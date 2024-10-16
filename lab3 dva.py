#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df = pd.read_csv('laptopData.csv')
print("Column Names in DataFrame:")
print(df.columns)
df.columns = df.columns.str.strip()
print("Original DataFrame:")
print(df.head())
if 'Price' in df.columns:
    df['Price_mean'] = df['Price'].fillna(df['Price'].mean())
    print("\nDataFrame after filling missing values with mean in column 'Price':")
    print(df[['Price', 'Price_mean']].head())
else:
    print("Column 'Price' not found!")
if 'RAM' in df.columns:
    df['RAM_median'] = df['RAM'].fillna(df['RAM'].median())
    print("\nDataFrame after filling missing values with median in column 'RAM':")
    print(df[['RAM', 'RAM_median']].head())
else:
    print("Column 'RAM' not found!")
if 'Weight' in df.columns:
    df['Weight_ffill'] = df['Weight'].fillna(method='ffill')
    print("\nDataFrame after forward filling missing values in column 'Weight':")
    print(df[['Weight', 'Weight_ffill']].head())
else:
    print("Column 'Weight' not found!")
if 'Weight' in df.columns:
    df['Weight_bfill'] = df['Weight'].fillna(method='bfill')
    print("\nDataFrame after backward filling missing values in column 'Weight':")
    print(df[['Weight', 'Weight_bfill']].head())
else:
    print("Column 'Weight' not found!")
scaler = MinMaxScaler()
if all(col in df.columns for col in ['Price', 'RAM', 'Weight']):
    df[['Price_minmax', 'RAM_minmax', 'Weight_minmax']] = scaler.fit_transform(df[['Price', 'RAM', 'Weight']])
    print("\nDataFrame after Min-Max Scaling:")
    print(df[['Price_minmax', 'RAM_minmax', 'Weight_minmax']].head())
else:
    print("One or more columns for Min-Max Scaling not found!")
scaler = StandardScaler()
if all(col in df.columns for col in ['Price', 'RAM', 'Weight']):
    df[['Price_zscore', 'RAM_zscore', 'Weight_zscore']] = scaler.fit_transform(df[['Price', 'RAM', 'Weight']])
    print("\nDataFrame after Z-score Normalization:")
    print(df[['Price_zscore', 'RAM_zscore', 'Weight_zscore']].head())
else:
    print("One or more columns for Z-score Normalization not found!")
print("\nFinal DataFrame:")
print(df.head())


# In[8]:


import pandas as pd
import numpy as np
df = pd.read_csv('laptopData.csv')
col = df['Price'][:10]
avg = np.mean(col)
xmin = np.min(col)
xmax = np.max(col)
for x in range(10):
    if np.isnan(df.loc[x, 'Price']):
        df.loc[x, 'Price'] = avg
    df.loc[x, 'Price'] = (df.loc[x, 'Price'] - xmin) / (xmax - xmin)
print(df['Price'][:10])


# In[ ]:


import pandas as pd
df=pd.read_csv('car-sales-extended-missing-data.csv')
col=df['Odometer (KM)'][:100]
col1=df['Doors'][:100]
col2=df['Make'][:100]
print(col2[:20])

import numpy as np
avg=np.mean(col)
std=np.std(col)
xmin=np.min(col)
print(std)
xmax=np.max(col)
x=0
for i in col:
    if(np.isnan(i)):
        df['Odometer (KM)'][x]=avg
    x+=1
print(col)

x=0
for i in col:
    df['Odometer (KM)'][x]=(df['Odometer (KM)'][x]-xmin)/(xmax-xmin)
    x+=1
print(col)

x=0
for i in col:
    df['Odometer (KM)'][x]=abs((df['Odometer (KM)'][x]-avg)/(std))
    x+=1
print(col)

median=np.median(col1)
x=0
for i in col1:
    if(np.isnan(i)):
        df['Doors'][x]=median
    x+=1
print(col1)

from scipy import stats
mode=st.mode(col2).mode[0]
x=0
for i in col2:
    if(isinstance(i, str)):
        continue
    if(np.isnan(i)):
        df['Make'][x]=mode
    x+=1
print(col2[:100])

