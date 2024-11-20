#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

survival_rate = titanic.groupby('pclass')['survived'].mean()

plt.figure(figsize=(8, 6))
survival_rate.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

survival_counts = titanic['survived'].value_counts()

plt.figure(figsize=(7, 7))
survival_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], 
                     labels=['Non-Survivors', 'Survivors'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Proportion of Survivors vs. Non-Survivors')
plt.ylabel('')
plt.show()

survival_by_class_and_sex = titanic.groupby(['pclass', 'sex', 'survived']).size().unstack(fill_value=0)

survival_by_class_and_sex.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'], figsize=(10, 6))
plt.title('Survival by Passenger Class and Sex')
plt.xlabel('Passenger Class and Sex')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0)
plt.legend(['Non-Survivors', 'Survivors'])
plt.show()

