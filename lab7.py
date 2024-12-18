#!/usr/bin/env python
# coding: utf-8

# In[6]:


#5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
file_path = 'Uber_pickup.csv'
uber_data = pd.read_csv(file_path)
uber_data['Datetime'] = pd.to_datetime(uber_data['Date'] + ' ' + uber_data['Time'])
uber_data['Day_of_Week'] = uber_data['Datetime'].dt.day_name()
uber_data['Hour'] = uber_data['Datetime'].dt.hour
uber_data['Region'] = uber_data['PU_Address'].str.extract(r'([A-Za-z ]+),')[0]
heatmap_data = uber_data.pivot_table(index='Day_of_Week', columns='Hour', aggfunc='size', fill_value=0)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(day_order)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
plt.title('Uber Pickups by Day of the Week and Hour', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Day of the Week', fontsize=12)
plt.show()
july_data = uber_data[uber_data['Datetime'].dt.month == 7]
july_pickup_trend = july_data.groupby(july_data['Datetime'].dt.date).size()
plt.figure(figsize=(14, 6))
july_pickup_trend.plot(kind='line', marker='o', color='blue', linestyle='-')
plt.title('Trend of Uber Pickups in July 2014', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Pickups', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
region_pickups = uber_data.groupby('Region').size().reset_index(name='Total_Pickups')
plt.figure(figsize=(12, 8))
plt.scatter(region_pickups['Region'], region_pickups['Total_Pickups'], 
            s=region_pickups['Total_Pickups'], alpha=0.6, color='teal', edgecolors='black')
plt.title('Number of Uber Pickups by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Number of Pickups', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[5]:


#3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COVID-19 daily dataset
file_path = 'day_wise.csv'  # Update this with your file path
data = pd.read_csv(file_path)

# Ensure proper datetime formatting
data['Date'] = pd.to_datetime(data['Date'])

# Task (a): Line Chart for Daily New Cases
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='New cases', label='Daily New Cases')
plt.title('Daily New COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()

# Task (b): Stacked Bar Chart for Confirmed, Recovered, and Fatal Cases
plt.figure(figsize=(12, 6))
plt.bar(data['Date'], data['Confirmed'], label='Confirmed', color='blue')
plt.bar(data['Date'], data['Recovered'], label='Recovered', color='green', bottom=data['Confirmed'] - data['Recovered'])
plt.bar(data['Date'], data['Deaths'], label='Deaths', color='red', bottom=data['Confirmed'] - data['Recovered'] - data['Deaths'])
plt.title('COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Task (c): Funnel Chart for Case Progression
latest_data = data.iloc[-1]  # Get the latest row
categories = ['Confirmed', 'Active', 'Recovered', 'Deaths']
values = [
    latest_data['Confirmed'],
    latest_data['Active'],
    latest_data['Recovered'],
    latest_data['Deaths']
]

plt.figure(figsize=(10, 6))
sns.barplot(x=values, y=categories, palette='viridis', orient='h')
plt.title('COVID-19 Case Progression Funnel')
plt.xlabel('Number of Cases')
plt.ylabel('Stage')
plt.show()


# In[14]:


pip install squarify


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify  # Make sure to install squarify

# Load the dataset
file_path = 'fifa21_male2.csv'  # Replace with your file path
fifa_data = pd.read_csv(file_path)

# Filter necessary columns and drop missing values
fifa_data_filtered = fifa_data[['OVA', 'Position', 'Club']].dropna()

# --- Restricting the range of players ---

# (a) Bar Chart: Compare the number of players across different positions
# Restricting to top 10 positions by number of players
position_counts = fifa_data_filtered['Position'].value_counts().head(10)

# Create bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=position_counts.index, y=position_counts.values, palette="viridis")
plt.title('Top 10 Positions by Number of Players', fontsize=16)
plt.xlabel('Position', fontsize=14)
plt.ylabel('Number of Players', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()

# (b) Donut Chart: Distribution of player ratings by position
# Define rating bins
bins = [0, 60, 70, 80, 90, 100]
labels = ['0-59', '60-69', '70-79', '80-89', '90-100']
fifa_data_filtered['Rating Range'] = pd.cut(fifa_data_filtered['OVA'], bins=bins, labels=labels)

# Count players by position and rating range, limit to top 5 positions
rating_distribution = fifa_data_filtered.groupby(['Position', 'Rating Range']).size().unstack()
rating_distribution = rating_distribution.loc[rating_distribution.sum(axis=1).sort_values(ascending=False).head(5).index]

# Create donut chart
rating_distribution.sum(axis=1).plot.pie(
    autopct='%1.1f%%', 
    startangle=140, 
    wedgeprops=dict(width=0.4), 
    figsize=(10, 8)
)
plt.title('Top 5 Position Player Rating Distribution', fontsize=16)
plt.ylabel('')
plt.show()

# (c) Tree Diagram: Hierarchical visualization of overall ratings by position and club
# Filter to keep only players with ratings above 70 (or any other criteria)
fifa_data_filtered_high_rating = fifa_data_filtered[fifa_data_filtered['OVA'] > 70]

# Prepare data for treemap
tree_data = fifa_data_filtered_high_rating.groupby(['Position', 'Club'])['OVA'].mean().reset_index()
tree_data['size'] = tree_data['OVA']

# Restrict to top 20 clubs based on average ratings
top_clubs = tree_data.groupby('Club')['size'].mean().sort_values(ascending=False).head(20).index
tree_data = tree_data[tree_data['Club'].isin(top_clubs)]

# Create tree map
plt.figure(figsize=(16, 10))
squarify.plot(
    sizes=tree_data['size'], 
    label=tree_data['Position'] + "\n" + tree_data['Club'], 
    alpha=0.8
)
plt.axis('off')
plt.title('Top 20 Clubs by Average Player Ratings (Above 70)', fontsize=16)
plt.show()


# In[ ]:




