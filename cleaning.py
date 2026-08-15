# Cleaning Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

np.random.seed(123)

# Load datasets
weather = pd.read_csv("data/4254465.csv")  
rides = pd.read_csv("data/fhvhv_tripdata_2026-01.csv")  

print(rides.head())
print(weather.head())
print(rides.columns)
print(rides.info())
print(weather.columns)
print(weather.info())

# create Sample data
rides_sample = rides.sample(n=100000, random_state=123)

# Missing value of ride data
missing_values = rides.isna().sum()
print(missing_values)

# Duplicate rows of ride data
print(rides_sample.duplicated().sum())

# outliers of ride data 
print(rides_sample['trip_miles'].describe())
print(rides_sample['base_passenger_fare'].describe())
print(rides_sample['trip_time'].describe())

## outliers:Boxplots
# Trip Distance
plt.figure()
sns.boxplot(y=rides_sample['trip_miles'])
plt.title("Outliers in Trip Distance")
plt.show()

# Fare
plt.figure()
sns.boxplot(y=rides_sample['base_passenger_fare'])
plt.title("Outliers in Fare")
plt.show()

# Trip Time
plt.figure()
sns.boxplot(y=rides_sample['trip_time'])
plt.title("Outliers in Trip Time")
plt.show()

## IQR method
# Trip distance
Q1_dist = rides_sample['trip_miles'].quantile(0.25)
Q3_dist = rides_sample['trip_miles'].quantile(0.75)
IQR_dist = Q3_dist - Q1_dist
lower_dist = Q1_dist - 1.5 * IQR_dist
upper_dist = Q3_dist + 1.5 * IQR_dist
print(lower_dist)
print(upper_dist)
print(((rides_sample['trip_miles'] < lower_dist) | (rides_sample['trip_miles'] > upper_dist)).sum())

# Fare
Q1_fare = rides_sample['base_passenger_fare'].quantile(0.25)
Q3_fare = rides_sample['base_passenger_fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare
print(lower_fare)
print(upper_fare)
print(((rides_sample['base_passenger_fare'] < lower_fare) | (rides_sample['base_passenger_fare'] > upper_fare)).sum())

# Trip Time
Q1_time = rides_sample['trip_time'].quantile(0.25)
Q3_time = rides_sample['trip_time'].quantile(0.75)
IQR_time = Q3_time - Q1_time
lower_time = Q1_time - 1.5 * IQR_time
upper_time = Q3_time + 1.5 * IQR_time
print(lower_time)
print(upper_time)
print(((rides_sample['trip_time'] < lower_time) | (rides_sample['trip_time'] > upper_time)).sum())

# Data Validation
print(rides_sample[
    (rides_sample['trip_miles'] <= 0) |
    (rides_sample['base_passenger_fare'] <= 0) |
    (rides_sample['trip_time'] <= 0)
].shape[0])

print(rides_sample[
    (rides_sample['trip_miles'] > 100) |
    (rides_sample['base_passenger_fare'] > 500)
].shape[0])

# Time consistency Check
print(rides_sample[rides_sample['pickup_datetime'] > rides_sample['dropoff_datetime']])

# Distance vs Fare
plt.figure()
plt.scatter(rides_sample['trip_miles'], rides_sample['base_passenger_fare'], alpha=0.3, color='blue')
plt.xlim(0, 200)
plt.ylim(0, 400)
plt.title("Fare vs Distance")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Base Passenger Fare")
plt.show()

# Distance vs Time
plt.figure()
plt.scatter(rides_sample['trip_miles'], rides_sample['trip_time'], alpha=0.3, color='green')
plt.title("Distance vs Trip Time")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Trip Time (seconds)")
plt.show()

# Time vs Fare
plt.figure()
plt.scatter(rides_sample['trip_time'], rides_sample['base_passenger_fare'], alpha=0.3, color='red')
plt.title("Fare vs Trip Time")
plt.xlabel("Trip Time (seconds)")
plt.ylabel("Base Passenger Fare")
plt.show()

# Distance vs Time vs Fare
pd.plotting.scatter_matrix(rides_sample[['trip_miles', 'trip_time', 'base_passenger_fare']])
plt.show()

print(rides_sample[
    (rides_sample['trip_miles'] <= 0) | (rides_sample['trip_time'] <= 0) | (rides_sample['base_passenger_fare'] <= 0)
])

rides_sample_speed = rides_sample.assign(speed_mph=rides_sample['trip_miles'] / (rides_sample['trip_time'] / 3600))
print(rides_sample_speed[rides_sample_speed['speed_mph'] > 100])

## datetime anamolies
rides_sample['pickup_datetime'] = pd.to_datetime(rides_sample['pickup_datetime'])
print(rides_sample[
    (rides_sample['pickup_datetime'] > pd.Timestamp.now()) |
    (rides_sample['pickup_datetime'] < pd.Timestamp("2025-12-31")) |
    (rides_sample['trip_time'] > 24 * 3600)
])

rides_clean = rides.copy()

rides_clean = rides_clean[
    (rides_clean['trip_miles'] > 0) &
    (rides_clean['trip_time'] > 0) &
    (rides_clean['base_passenger_fare'] > 0)
]

# Handling missing values
rides_clean = rides_clean[~rides_clean['originating_base_num'].isna()]
rides_clean = rides_clean.assign(speed_mph=rides_clean['trip_miles'] / (rides_clean['trip_time'] / 3600))
rides_clean = rides_clean[rides_clean['speed_mph'] <= 100]
rides_clean = rides_clean.drop(columns=['speed_mph'])

# outliers: IQR method
# Trip distance
Q1_dist = rides_clean['trip_miles'].quantile(0.25)
Q3_dist = rides_clean['trip_miles'].quantile(0.75)
IQR_dist = Q3_dist - Q1_dist
rides_clean = rides_clean[
    (rides_clean['trip_miles'] >= (Q1_dist - 1.5 * IQR_dist)) &
    (rides_clean['trip_miles'] <= (Q3_dist + 1.5 * IQR_dist))
]
# Trip Time
Q1_time = rides_clean['trip_time'].quantile(0.25)
Q3_time = rides_clean['trip_time'].quantile(0.75)
IQR_time = Q3_time - Q1_time
rides_clean = rides_clean[
    (rides_clean['trip_time'] >= (Q1_time - 1.5 * IQR_time)) &
    (rides_clean['trip_time'] <= (Q3_time + 1.5 * IQR_time))
]

# Fare
Q1_fare = rides_clean['base_passenger_fare'].quantile(0.25)
Q3_fare = rides_clean['base_passenger_fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
rides_clean = rides_clean[
    (rides_clean['base_passenger_fare'] >= (Q1_fare - 1.5 * IQR_fare)) &
    (rides_clean['base_passenger_fare'] <= (Q3_fare + 1.5 * IQR_fare))
]

# datetime anomalies
rides_clean['pickup_datetime'] = pd.to_datetime(rides_clean['pickup_datetime'])
rides_clean = rides_clean[
    (rides_clean['pickup_datetime'] >= pd.Timestamp("2026-01-01")) &
    (rides_clean['pickup_datetime'] <= pd.Timestamp.now()) &
    (rides_clean['trip_time'] <= 24 * 3600)
]

# save the ride cleaned data
rides_clean.to_csv("data/rides_clean.csv", index=False)

# check for cleaned ride data
print("original dataset: ", rides.shape[0], "\n")
print("cleaned dataset : ", rides_clean.shape[0], "\n\n")

zero_negative_check = rides_clean[
    (rides_clean['trip_miles'] <= 0) | (rides_clean['trip_time'] <= 0) | (rides_clean['base_passenger_fare'] <= 0)
]
print("Trips with zero or negative values are ", zero_negative_check.shape[0], "\n\n")

missing_originating = rides_clean['originating_base_num'].isna().sum()
print("missing values are ", missing_originating, "\n\n")

unrealistic_speeds = rides_clean.assign(speed_mph=rides_clean['trip_miles'] / (rides_clean['trip_time'] / 3600))
unrealistic_speeds = unrealistic_speeds[unrealistic_speeds['speed_mph'] > 100]
print("Trips with unrealistic speed are ", unrealistic_speeds.shape[0], "\n\n")

datetime_anomalies = rides_clean[
    (rides_clean['pickup_datetime'] < pd.Timestamp("2026-01-01")) |
    (rides_clean['pickup_datetime'] > pd.Timestamp.now()) |
    (rides_clean['trip_time'] > 24 * 3600)
]
print("Trips with datetime anomalies are", datetime_anomalies.shape[0], "\n\n")

print("Summary of Trip distance")
print(rides['trip_miles'].describe())
print("\n")
print(rides_clean['trip_miles'].describe())
print("\n\n")
print("Summary of Trip Time")
print(rides['trip_time'].describe())
print("\n")
print(rides_clean['trip_time'].describe())
print("\n\n")
print("Summary of base Fare")
print(rides['base_passenger_fare'].describe())
print("\n")
print(rides_clean['base_passenger_fare'].describe())
print("\n\n")

# Plots
# Trip Distance
plt.figure()
sns.boxplot(y=rides_clean['trip_miles'], color="skyblue")
plt.title("Trip Miles(Cleaned)")
plt.show()

# Trip Time
plt.figure()
sns.boxplot(y=rides_clean['trip_time'], color="lightgreen")
plt.title("Trip Time(Cleaned)")
plt.show()

# Fare
plt.figure()
sns.boxplot(y=rides_clean['base_passenger_fare'], color="salmon")
plt.title("Base Fare(Cleaned)")
plt.show()

## Weather Data:
# Missing values
missing_weather = weather.isna().sum()
print(missing_weather)

# Duplicate rows
print(weather.duplicated().sum())

# Data validation
print(weather[(weather['TMIN'] < -30) | (weather['TMAX'] > 110) | (weather['TAVG'] < -30) | (weather['TAVG'] > 110)])
print(weather[(weather['PRCP'] < 0) | (weather['SNOW'] < 0) | (weather['SNWD'] < 0)])
print(weather[(weather['AWND'] > 100) | (weather['WSF2'] > 100) | (weather['WSF5'] > 100)])

# Range Checks
print((weather['DATE'].min(), weather['DATE'].max()))

# Categorical / attribute columns
attribute_cols = [c for c in weather.columns if "ATTRIBUTES" in c]
for col in attribute_cols:
    print(weather[col].unique())

# Weather Data Cleaning
weather_clean = weather.copy()
weather_clean = weather_clean[[c for c in weather_clean.columns if not c.endswith("_ATTRIBUTES")]]

# Handling missing values
wind_cols = ["AWND", "WDF2", "WDF5", "WSF2", "WSF5"]
for col in wind_cols:
    median_val = weather_clean[col].median()
    weather_clean[col] = weather_clean[col].fillna(median_val)

weather_clean = weather_clean[~weather_clean['DATE'].isna() & ~weather_clean['STATION'].isna()]

weather_clean = weather_clean[
    (weather_clean['TMIN'] >= -30) & (weather_clean['TMAX'] <= 110) &
    (weather_clean['TAVG'] >= -30) & (weather_clean['TAVG'] <= 110)
]
weather_clean = weather_clean[
    (weather_clean['PRCP'] >= 0) & (weather_clean['SNOW'] >= 0) & (weather_clean['SNWD'] >= 0)
]
weather_clean = weather_clean[
    (weather_clean['AWND'] <= 100) & (weather_clean['WSF2'] <= 100) & (weather_clean['WSF5'] <= 100)
]

weather_clean['DATE'] = pd.to_datetime(weather_clean['DATE'])
weather_clean = weather_clean[
    (weather_clean['DATE'] >= pd.Timestamp("2026-01-01")) & (weather_clean['DATE'] <= pd.Timestamp("2026-01-31"))
]

print(weather_clean.describe(include='all'))

# save the cleaned weather data
weather_clean.to_csv("data/weather_clean.csv", index=False)

rides_clean = pd.read_csv("data/rides_clean.csv")
weather_clean = pd.read_csv("data/weather_clean.csv")

# create date column in ride
rides_clean['pickup_datetime'] = pd.to_datetime(rides_clean['pickup_datetime'])
rides_clean['DATE'] = rides_clean['pickup_datetime'].dt.date

# merge ride with weather
weather_clean['DATE'] = pd.to_datetime(weather_clean['DATE']).dt.date
rides_weather = rides_clean.merge(weather_clean, how='left', on='DATE')
print(rides_clean.shape)
print(rides_weather.shape)
print(rides_weather['TAVG'].isna().sum())
rides_weather.to_csv("data/rides_weather_merged.csv", index=False)