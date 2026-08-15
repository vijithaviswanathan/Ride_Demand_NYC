import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

model_data = pd.read_csv(os.path.join("analysis_data", "model_data.csv"))
print(model_data.columns)

# Convert categorical variables to factors
model_data['rain_flag'] = model_data['rain_flag'].astype('category')
model_data['snow_flag'] = model_data['snow_flag'].astype('category')
model_data['is_weekend'] = model_data['is_weekend'].astype('category')

# Regression model
demand_model = smf.ols(
    "ride_count ~ avg_fare + avg_trip_miles + avg_trip_time + "
    "rain_flag + snow_flag + is_weekend + hour_sin + hour_cos",
    data=model_data
).fit()

print(demand_model.summary())

# Average demand by hour
plt.figure()
sns.lineplot(data=model_data, x='hour', y='ride_count', estimator='mean', errorbar=None)
plt.title("Average Ride Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Ride Count")
plt.show()

# weekend vs weekday
plt.figure()
sns.boxplot(data=model_data, x='is_weekend', y='ride_count', hue='is_weekend',
            palette=["skyblue", "orange"])
plt.title("Ride Demand: Weekend vs Weekday")
plt.xlabel("Weekend (0 = Weekday, 1 = Weekend)")
plt.ylabel("Ride Count")
plt.show()

# rain vs no rain
plt.figure()
sns.boxplot(data=model_data, x='rain_flag', y='ride_count', hue='rain_flag', palette="Blues")
plt.title("Ride Demand: Rain vs No Rain")
plt.xlabel("Rain Condition")
plt.ylabel("Ride Count")
plt.show()

# snow vs no snow
plt.figure()
sns.boxplot(data=model_data, x='snow_flag', y='ride_count', hue='snow_flag', palette="Greys")
plt.title("Ride Demand: Snow vs No Snow")
plt.xlabel("Snow Condition")
plt.ylabel("Ride Count")
plt.show()

# fare vs ride demand
plt.figure()
plt.scatter(model_data['avg_fare'], model_data['ride_count'], alpha=0.3, color='red')
sns.regplot(data=model_data, x='avg_fare', y='ride_count', scatter=False, color='black')
plt.title("Relationship Between Fare and Ride Demand")
plt.xlabel("Average Fare")
plt.ylabel("Ride Count")
plt.show()