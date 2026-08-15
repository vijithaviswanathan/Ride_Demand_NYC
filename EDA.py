# EDA 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

np.random.seed(123)

rides_weather = pd.read_csv("data/rides_weather_merged.csv")
rides_weather['pickup_datetime'] = pd.to_datetime(rides_weather['pickup_datetime'])
rides_weather['DATE'] = pd.to_datetime(rides_weather['DATE'])

# Sample Data
sample_data = rides_weather.sample(n=min(1000000, len(rides_weather)), random_state=123)

# Feature Engineering
sample_data = sample_data.assign(
    hour=sample_data['pickup_datetime'].dt.hour,
    day=sample_data['pickup_datetime'].dt.date,
    day_of_week=sample_data['pickup_datetime'].dt.strftime('%a'),
)
sample_data['is_weekend'] = np.where(sample_data['day_of_week'].isin(["Sat", "Sun"]), 1, 0)
sample_data['rain_flag'] = np.where(sample_data['PRCP'] > 0, "Rain", "No_Rain")
sample_data['snow_flag'] = np.where(sample_data['SNOW'] > 0, "Snow", "No_Snow")

print("Summary Statistics of sample data is")
print(sample_data[["trip_miles", "trip_time", "base_passenger_fare"]].describe())

# Daily Demand 
daily_rides = sample_data.groupby('day').size().reset_index(name='ride_count')
print(daily_rides['ride_count'].describe())

# Daily ride distribution
plt.figure()
sns.histplot(daily_rides['ride_count'], bins=30, color="skyblue")
plt.title("Daily ride demand distribution")
plt.show()

# Demand vs Price
price_demand = sample_data.copy()
price_demand['fare_bin'] = pd.cut(price_demand['base_passenger_fare'], bins=range(0, 55, 5))
price_demand = price_demand.groupby('fare_bin').size().reset_index(name='ride_count')

#  Weather Impact Analysis
# Rain impact
rain_summary = sample_data.groupby(['day', 'rain_flag']).agg(
    ride_count=('rain_flag', 'size'),
    avg_trip_time=('trip_time', 'mean'),
    avg_fare=('base_passenger_fare', 'mean')
).reset_index()
rain_summary = rain_summary.groupby('rain_flag').agg(
    avg_daily_rides=('ride_count', 'mean'),
    avg_trip_time=('avg_trip_time', 'mean'),
    avg_fare=('avg_fare', 'mean')
).reset_index()

# Snow impact
snow_summary = sample_data.groupby(['day', 'snow_flag']).agg(
    ride_count=('snow_flag', 'size'),
    avg_trip_time=('trip_time', 'mean'),
    avg_fare=('base_passenger_fare', 'mean')
).reset_index()
snow_summary = snow_summary.groupby('snow_flag').agg(
    avg_daily_rides=('ride_count', 'mean'),
    avg_trip_time=('avg_trip_time', 'mean'),
    avg_fare=('avg_fare', 'mean')
).reset_index()

# Correlation matrix
numeric_vars = sample_data[['trip_miles', 'trip_time', 'base_passenger_fare', 'TAVG', 'PRCP', 'SNOW']]
cor_matrix = numeric_vars.corr()

print("Correlation Matrix")
print(cor_matrix)

plt.figure()
sns.boxplot(y=sample_data['trip_miles'])
plt.title("Trip Distance (miles)")
plt.show()

plt.figure()
sns.boxplot(y=sample_data['trip_time'])
plt.title("Trip Time (seconds)")
plt.show()

plt.figure()
sns.boxplot(y=sample_data['base_passenger_fare'])
plt.title("Fare ($)")
plt.show()

sample_data = sample_data.assign(
    hour=sample_data['pickup_datetime'].dt.hour,
    day=sample_data['pickup_datetime'].dt.date,
    day_of_week=sample_data['pickup_datetime'].dt.strftime('%a'),
)
sample_data['is_weekend'] = np.where(sample_data['day_of_week'].isin(["Sat", "Sun"]), 1, 0)
sample_data['month'] = sample_data['pickup_datetime'].dt.month
sample_data['day_of_month'] = sample_data['pickup_datetime'].dt.day
sample_data['hour_sin'] = np.sin(2 * np.pi * sample_data['hour'] / 24)
sample_data['hour_cos'] = np.cos(2 * np.pi * sample_data['hour'] / 24)

# Holiday Flag
nyc_holidays = pd.to_datetime(["2026-01-01", "2026-07-04", "2026-12-25"]).date
sample_data['is_holiday'] = np.where(sample_data['day'].isin(nyc_holidays), 1, 0)

# Weather Feature
sample_data['PRCP'] = sample_data['PRCP'].fillna(0)
sample_data['SNOW'] = sample_data['SNOW'].fillna(0)
sample_data['TAVG'] = sample_data['TAVG'].fillna(sample_data['TAVG'].mean())
sample_data['rain_flag'] = np.where(sample_data['PRCP'] > 0, "Rain", "No_Rain")
sample_data['snow_flag'] = np.where(sample_data['SNOW'] > 0, "Snow", "No_Snow")

# Trip Feature
sample_data['fare_per_mile'] = np.where(sample_data['trip_miles'] > 0,
                                         sample_data['base_passenger_fare'] / sample_data['trip_miles'], np.nan)
sample_data['speed_mph'] = np.where(sample_data['trip_time'] > 0,
                                     sample_data['trip_miles'] / (sample_data['trip_time'] / 3600), np.nan)

# Handling outliers
sample_data = sample_data[
    (sample_data['trip_miles'] > 0) &
    (sample_data['trip_time'] > 0) &
    (sample_data['base_passenger_fare'] > 0) &
    (sample_data['fare_per_mile'] > 0) &
    (sample_data['fare_per_mile'] < 100) &
    (sample_data['speed_mph'] > 1) & (sample_data['speed_mph'] < 80)
]

# Distance and Fare
sample_data['trip_distance'] = sample_data['trip_miles']
sample_data['fare_distance_interaction'] = sample_data['fare_per_mile'] * sample_data['trip_miles']

# Peak hour indicators
sample_data['morning_peak'] = np.where(sample_data['hour'].isin(range(7, 10)), 1, 0)
sample_data['evening_peak'] = np.where(sample_data['hour'].isin(range(17, 21)), 1, 0)
sample_data['late_night'] = np.where(sample_data['hour'].isin(range(0, 6)), 1, 0)

hourly_data = sample_data.groupby(['day', 'hour']).agg(
    ride_count=('hour', 'size'),
    avg_fare=('base_passenger_fare', 'mean'),
    avg_trip_miles=('trip_miles', 'mean'),
    avg_trip_time=('trip_time', 'mean'),
    avg_speed=('speed_mph', 'mean'),
    rain_flag=('rain_flag', 'first'),
    snow_flag=('snow_flag', 'first'),
    is_weekend=('is_weekend', 'first'),
    is_holiday=('is_holiday', 'first'),
    month=('month', 'first'),
    hour_sin=('hour_sin', 'first'),
    hour_cos=('hour_cos', 'first'),
).reset_index()
hourly_data = hourly_data.sort_values(['day', 'hour'])

# High demand flag
threshold = hourly_data['ride_count'].quantile(0.9)
hourly_data['high_demand'] = np.where(hourly_data['ride_count'] >= threshold, 1, 0)

# Lag and rolling feature
hourly_data = hourly_data.sort_values(['day', 'hour'])
hourly_data['lag_1hr'] = hourly_data['ride_count'].shift(1)
hourly_data['lag_24hr'] = hourly_data['ride_count'].shift(24)
hourly_data['rolling_3hr'] = hourly_data['ride_count'].rolling(window=3).mean()
model_data = hourly_data.dropna()

# Join hourly demand to trip level data
sample_data = sample_data.merge(
    hourly_data[['day', 'hour', 'ride_count']],
    how='left', on=['day', 'hour']
)

# Convert categorical variables into factors
sample_data['rain_flag'] = sample_data['rain_flag'].astype('category')
sample_data['snow_flag'] = sample_data['snow_flag'].astype('category')
sample_data['day_of_week'] = sample_data['day_of_week'].astype('category')
sample_data['is_weekend'] = sample_data['is_weekend'].astype('category')
sample_data['is_holiday'] = sample_data['is_holiday'].astype('category')

# Interaction features
sample_data['rain_peak_interaction'] = (sample_data['rain_flag'] == "Rain").astype(float) * sample_data['hour']
sample_data['weekend_peak'] = (sample_data['is_weekend'].astype(str) == "1").astype(float) * sample_data['hour']

print(sample_data.info())

print("Summary of sample data:")
print(sample_data[["trip_miles", "trip_time", "base_passenger_fare",
                    "fare_per_mile", "speed_mph", "ride_count"]].describe())

print(sample_data.shape, "\n")
print(model_data.shape, "\n")

print(model_data.info())

print(model_data[['ride_count', 'avg_fare', 'avg_trip_miles', 'avg_trip_time', 'avg_speed']].describe())

# Ride demand by hour plot
plt.figure()
plt.plot(model_data['hour'], model_data['ride_count'], color='blue')
plt.scatter(model_data['hour'], model_data['ride_count'])
plt.title("Ride Demand by Hour")
plt.xlabel("Hour")
plt.ylabel("Ride Count")
plt.show()

# Weekend vs Weekday
weekly_summary = model_data.groupby('is_weekend').agg(total_rides=('ride_count', 'sum')).reset_index()

plt.figure()
sns.barplot(data=weekly_summary, x=weekly_summary['is_weekend'].astype(str), y='total_rides',
            hue=weekly_summary['is_weekend'].astype(str))
plt.title("Weekend vs Weekday Ride Demand")
plt.xlabel("Weekend (1 = Yes)")
plt.ylabel("Total Rides")
plt.show()

# Weather Impact
# Rain vs No Rain
rain_summary = model_data.groupby('rain_flag').agg(
    avg_hourly_rides=('ride_count', 'mean'),
    avg_fare=('avg_fare', 'mean')
).reset_index()

plt.figure()
sns.barplot(data=rain_summary, x='rain_flag', y='avg_hourly_rides', hue='rain_flag')
plt.title("Average Hourly Rides (Rain vs No Rain)")
plt.show()

# Snow vs No Snow
snow_summary = model_data.groupby('snow_flag').agg(
    avg_hourly_rides=('ride_count', 'mean'),
    avg_fare=('avg_fare', 'mean')
).reset_index()

plt.figure()
sns.barplot(data=snow_summary, x='snow_flag', y='avg_hourly_rides', hue='snow_flag')
plt.title("Average Hourly Rides(Snow vs No Snow)")
plt.show()

# Price vs hourly demand
plt.figure()
plt.scatter(model_data['avg_fare'], model_data['ride_count'], alpha=0.5, color='red')
plt.title("Price vs Hourly Ride Demand")
plt.xlabel("Average Fare")
plt.ylabel("Ride Count")
plt.show()

# Linear regression
price_model = smf.ols("ride_count ~ avg_fare", data=model_data).fit()
print(price_model.summary())

# Surge detection
model_data['surge_flag'] = np.where(model_data['ride_count'] > 1.2 * model_data['ride_count'].mean(), "Surge", "Normal")

plt.figure()
sns.scatterplot(data=model_data, x='hour', y='ride_count', hue='surge_flag')
plt.title("Hourly Surge Detection")
plt.show()

# Feature checks
plt.figure()
sns.histplot(model_data['avg_speed'], bins=50, color="lightgreen")
plt.title("Average Speed Distribution")
plt.show()

# Correlation Matrix
numeric_vars = model_data[['ride_count', 'avg_fare', 'avg_trip_miles', 'avg_trip_time', 'avg_speed']]
cor_matrix = numeric_vars.corr()
print(cor_matrix)

import os
if not os.path.exists("analysis_data"):
    os.makedirs("analysis_data")
model_data.to_csv("analysis_data/model_data.csv", index=False)