#!/usr/bin/env python
# coding: utf-8

# In[412]:


#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
fig=go.Figure()


# In[413]:


#load the dataset
data=pd.read_csv('rideshare_kaggle.csv')


# In[414]:


#check the first 5 records
data.head()


# In[415]:


#the last 5 records
data.tail()


# In[416]:


#columns
data.columns


# In[417]:


data.shape


# In[418]:


#description of the dataset
data.info()


# In[419]:


#statistical summary of the numerical columns
data.describe()


# In[420]:


#statistical summary of the object columns
data.describe(include='object')


# In[421]:


#data types of the features
data.dtypes


# In[422]:


#unique value in each columns
data.nunique()


# In[423]:


#total count of occurrences for each category in source columns
data['source'].value_counts()


# In[424]:


#total count of occurrences for each category in cab columns
data.cab_type.value_counts()


# In[425]:


#total count of occurrences for each category in car name columns
data.name.value_counts()


# In[426]:


#Average of prices for cab type with its name
data.groupby(['cab_type', 'name'])['price'].mean().sort_values(ascending=False)


# In[427]:


#total sum of surge multiplier for cab type with its source

data.groupby(['cab_type', 'source'])['surge_multiplier'].sum().sort_values(ascending=False)


# In[428]:


#calculates average distance with cab type and name
avg_distance = data.groupby(['cab_type', 'name'])['distance'].mean().reset_index()
avg_distance


# In[429]:


#conversion to date time format
data['datetime']=pd.to_datetime(data['datetime'])


# In[430]:


#check the numerical columns
numerical_cols=data.columns[data.dtypes !='object']
print(numerical_cols)
len(data._get_numeric_data().columns)


# In[431]:


#check the categorical columns in  the dataset
categorical_cols=data.columns[data.dtypes =='object']
print(categorical_cols)


# In[432]:


uber_data = data[data['cab_type'] == 'Uber']

# Find the maximum price for Uber rides
max_price_uber = uber_data['price'].max()

# Filter data for Lyft rides
lyft_data = data[data['cab_type'] == 'Lyft']

# Find the maximum price for Lyft rides
max_price_lyft = lyft_data['price'].max()

print("Maximum price for Uber rides:", max_price_uber)
print("Maximum price for Lyft rides:", max_price_lyft)


# # Data Cleaning
Missing Values

# In[433]:


#checks the null values and returns in Boolean format
data.isnull().values.any()


# In[434]:


#calculates the total missing value and its percentage
total = data.isnull().sum().sort_values(ascending=False)
percentage= data.isnull().sum()/data.isnull().count()*100
missing_data = pd.concat([total, percentage], axis=1, keys=['Total Missing Values', 'Percentage'])
missing_data


# In[435]:


#imputation method 
median_price = data['price'].median()

# Fill missing values in the 'price' column with the median
data['price'].fillna(value=median_price, inplace=True)
median_price


# In[436]:


#to check if the missing values have been successfully filled
missing_values = data['price'].isnull().sum()
missing_values


# In[437]:


#sum of missing values in each columns
data.isna().sum()

Duplicate Values
# In[438]:


#Total sum of duplicate values
print("Number of duplicates:",data.duplicated().sum())

Handle Outliers
# In[439]:


# Select only numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(10, 5, figsize=(15, 20))
axes = axes.flatten()
# Creating boxplot
for i, column in enumerate(numerical_cols):
    axes[i].boxplot(data[column])
    axes[i].set_title(column)
    
# Removing the unused plots
for j in range(len(numerical_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()

plt.show() 


# In[440]:


def remove_outliers_z_score(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    outliers = np.abs(z_scores) > threshold
    data = data[~outliers]
    return df


# In[441]:


data.describe()


# In[442]:


df.describe()


# In[443]:


#IQR method to drop outliers
def drop_outliers(data, field_name):
    iqr = 1.5 * (np.percentile(data[field_name], 75) - np.percentile(data[field_name], 25))
    data.drop(data[data[field_name] > (iqr + np.percentile(data [field_name], 75)) ].index, inplace=True) 
    data.drop(data[data[field_name] < (np.percentile(data [field_name], 25) - iqr)].index, inplace=True)


# In[444]:


#boxplot
fig, axes = plt.subplots(10, 5, figsize=(15, 20))
axes = axes.flatten()
# Creating boxplot
for i, column in enumerate(numerical_cols):
    axes[i].boxplot(data[column])
    axes[i].set_title(column)
    
# Removing the unused plots
for j in range(len(numerical_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()

plt.show()    #found outliers  in price,distance,surge_multiplier,latitude,temperature,apparentTemperature,precipIntensity,precipProbability,windGust,visibility,visibility.1,temperatureHigh,temperatureMax,temperatureMin,uvIndex,apparentTemperatureMax,apparentTemperatureMin


# In[445]:


#check if both columns are same
(data['temperatureHigh'] == data['temperatureMax']).all()


# In[446]:


#check if both columns are same
(data['visibility'] == data['visibility.1']).all()


# In[447]:


#check if both columns are same
(data['apparentTemperatureHigh'] == data['apparentTemperatureMax']).all()


# In[448]:


#correlation of all columns with price
d=data.corr()['price'].sort_values(ascending=False)
d


# In[449]:


f_corr = ['temperature','apparentTemperature','temperatureHigh','temperatureLow','apparentTemperatureHigh',
            'apparentTemperatureLow','temperatureMin','temperatureHighTime','temperatureMax',
            'apparentTemperatureMin','apparentTemperatureMax','price']
cor = data[f_corr].corr()


# In[450]:


#heatmap
plt.figure(figsize=(12,12))
sns.heatmap(cor,annot=True,cmap='coolwarm')


# In[451]:


#drop columns with less correlation
df= data.drop(['temperature','apparentTemperature','temperatureHigh','temperatureLow','apparentTemperatureHigh',
                'apparentTemperatureLow','temperatureMin','temperatureHighTime','temperatureMax',
                      'apparentTemperatureMin','apparentTemperatureMax','id', 'datetime', 'timezone','long_summary','timestamp'],axis=1)
df.shape


# In[452]:


f_corr =['precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'visibility', 'dewPoint', 'pressure', 'windBearing',
       'cloudCover', 'uvIndex', 'ozone', 'moonPhase',
       'precipIntensityMax','surge_multiplier','price']

cor = data[f_corr].corr()
plt.figure(figsize=(12,12))
sns.heatmap(cor,annot=True,cmap='coolwarm')


# In[453]:


df = df.drop(['precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'visibility', 'dewPoint', 'pressure', 'windBearing',
       'cloudCover', 'uvIndex', 'ozone', 'moonPhase',
       'precipIntensityMax','windGustTime',
       'temperatureLowTime', 'apparentTemperatureHighTime',
       'apparentTemperatureLowTime','sunriseTime',
       'sunsetTime', 'uvIndexTime', 'temperatureMinTime', 'temperatureMaxTime',
       'apparentTemperatureMinTime', 'apparentTemperatureMaxTime'],axis=1)
df.shape


# In[454]:


#Columns after dropping unneccessary columns
df.info()
df.shape


# # DATA VISUALIZATION

# In[455]:


#histogram to chevk the count of cab type by hour
fig = px.histogram(df, x='hour', color='cab_type')

# Update layout
fig.update_layout(
    title='Count of rides by hour and cab type',
    xaxis_title='Hour',
    yaxis_title='Count'
)

# Show the plot
fig.show()


# In[456]:


#pie chart showing count of sources
fig = px.pie(df, names='source', title='Count of Source', color='source')
fig.update_layout(xaxis_title='Source', yaxis_title='Count')
fig.show()


# In[457]:


#line plot of price vs distance
x_values = df['day']
y_values = df['price']

# Create line plot
sns.lineplot(x=x_values, y=y_values)

# Add labels and title
plt.xlabel('Distance')
plt.ylabel('Price')
plt.title('Line Plot of Price vs Distance')


# In[458]:


average_prices = data.groupby('cab_type')['price'].mean().reset_index()

# Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=average_prices, x='cab_type', y='price', palette='muted')

# Add labels and title
plt.xlabel('Cab Type')
plt.ylabel('Average Price')
plt.title('Average Price of Each Cab Type')

# Show plot
plt.show()


# In[459]:


x_values = df['distance']          # Continuous column for x-axis
y_values = df['surge_multiplier']  # Continuous column for y-axis
hue_values = df['cab_type']        # Categorical column for hue

# Create line plot with hue
sns.lineplot(x=x_values, y=y_values, hue=hue_values)

# Add labels and title
plt.xlabel('Distance')
plt.ylabel('Surge Multiplier')
plt.title('Line Plot of Surge Multiplier over Distance with Cab Type')

# Show legend
plt.legend(title='Cab Type')

# Show plot
plt.show()


# In[460]:


df.corr()
sns.heatmap(df.corr(),annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)


# # Data Encoding

# In[461]:


categorical_cols=df.columns[df.dtypes =='object']
print(categorical_cols)


# In[462]:


df['destination'].unique()


# In[463]:


df['cab_type'].unique()


# In[464]:


df['name'].unique()


# In[465]:


df['icon'].unique()


# In[466]:


from sklearn.preprocessing import LabelEncoder
l_encoder=LabelEncoder()


# In[467]:


for i in categorical_cols:
    df[i] = l_encoder.fit_transform(df[i])


# In[468]:


df


# In[469]:


plt.figure(figsize=(9,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f",cmap="coolwarm")
plt.show()


# # Model Creation

# In[470]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression

X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)


print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root mean square Error: {rmse}')


# In[471]:


#scales and transforms the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[472]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


poly = PolynomialFeatures(degree=2)  
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Initialize and train the Ridge Regression model (with regularization)
ridge_model = Ridge(alpha=0.1)  # You can experiment with different alpha values
ridge_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = ridge_model.predict(X_test_scaled)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)


# In[473]:


# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the RandomForestRegressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# In[ ]:




