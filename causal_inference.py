import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

rides_weather = pd.read_csv(os.path.join("data", "rides_weather_merged.csv"))

rides_weather['pickup_datetime'] = pd.to_datetime(rides_weather['pickup_datetime'])
rides_weather['date'] = rides_weather['pickup_datetime'].dt.date
rides_weather['hour'] = rides_weather['pickup_datetime'].dt.hour
rides_weather['day_of_week'] = rides_weather['pickup_datetime'].dt.strftime('%a')
rides_weather['is_weekend'] = np.where(rides_weather['day_of_week'].isin(["Sat", "Sun"]), 1, 0)
rides_weather['rain_flag'] = np.where(rides_weather['PRCP'] > 0, 1, 0)
rides_weather['snow_flag'] = np.where(rides_weather['SNOW'] > 0, 1, 0)

# Aggregate to hourly level
hourly_data = rides_weather.groupby(['date', 'hour']).agg(
    ride_count=('hour', 'size'),
    avg_fare=('base_passenger_fare', 'mean'),
    avg_trip_miles=('trip_miles', 'mean'),
    avg_trip_time=('trip_time', 'mean'),
    TAVG=('TAVG', 'first'),
    rain_flag=('rain_flag', 'first'),
    snow_flag=('snow_flag', 'first'),
    is_weekend=('is_weekend', 'first'),
).reset_index()
hourly_data = hourly_data.dropna(subset=['ride_count', 'avg_fare', 'hour', 'is_weekend', 'TAVG', 'rain_flag', 'snow_flag'])

hourly_summary = pd.DataFrame({
    "total_hours": [len(hourly_data)],
    "avg_hourly_demand": [hourly_data['ride_count'].mean()],
    "mean_fare": [hourly_data['avg_fare'].mean()],
    "min_fare": [hourly_data['avg_fare'].min()],
    "max_fare": [hourly_data['avg_fare'].max()],
    "sd_fare": [hourly_data['avg_fare'].std()],
})

print(hourly_summary.round(3))

# Plots
plt.figure()
sns.regplot(data=hourly_data, x='avg_fare', y='ride_count', scatter_kws={"alpha": 0.5})
plt.title("Baseline Relationship Between Price and Ride Demand")
plt.xlabel("Average Hourly Fare")
plt.ylabel("Hourly Ride Count")
plt.show()

# Baseline regression model
baseline_price_model = smf.ols(
    "ride_count ~ avg_fare + hour + is_weekend + TAVG + rain_flag + snow_flag + "
    "avg_trip_miles + avg_trip_time",
    data=hourly_data
).fit()

baseline_results = baseline_price_model.summary2().tables[1]
baseline_results['conf_int_lower'] = baseline_price_model.conf_int()[0]
baseline_results['conf_int_upper'] = baseline_price_model.conf_int()[1]

baseline_price_effect = baseline_results.loc[['avg_fare']]

print(baseline_price_effect.round(4))

# Propensity Score Matching
fare_median = hourly_data['avg_fare'].median()

hourly_data['high_price'] = np.where(hourly_data['avg_fare'] > fare_median, 1, 0)

ps_formula = "high_price ~ hour + is_weekend + TAVG + rain_flag + snow_flag + avg_trip_miles + avg_trip_time"
ps_model = smf.logit(ps_formula, data=hourly_data).fit()

hourly_data['propensity_score'] = ps_model.predict(hourly_data)
hourly_data['logit_ps'] = np.log(hourly_data['propensity_score'] / (1 - hourly_data['propensity_score']))

treated = hourly_data[hourly_data['high_price'] == 1]
control = hourly_data[hourly_data['high_price'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['logit_ps']])
distances, indices = nn.kneighbors(treated[['logit_ps']])

matched_control = control.iloc[indices.flatten()].copy()
matched_treated = treated.copy()
matched_treated['weights'] = 1
matched_control['weights'] = 1

matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)

# Blanace Check
print(ps_model.summary())

matched_price_model = smf.wls(
    "ride_count ~ high_price",
    data=matched_data,
    weights=matched_data['weights']
).fit()

print(matched_price_model.summary())

matched_price_effect = matched_price_model.summary2().tables[1]
matched_price_effect['conf_int_lower'] = matched_price_model.conf_int()[0]
matched_price_effect['conf_int_upper'] = matched_price_model.conf_int()[1]
matched_price_effect = matched_price_effect.loc[['high_price']]

print(matched_price_effect)

# Load EM cluster data
model_data_em = pd.read_csv(os.path.join("analysis_data", "model_data_em.csv"))

model_data_em['rain_flag'] = model_data_em['rain_flag'].astype('category')
model_data_em['snow_flag'] = model_data_em['snow_flag'].astype('category')
model_data_em['is_weekend'] = model_data_em['is_weekend'].astype('category')
model_data_em['cluster'] = model_data_em['cluster'].astype('category')

# Cluster summary
cluster_summary = model_data_em.groupby('cluster').agg(
    avg_demand=('ride_count', 'mean'),
    avg_fare=('avg_fare', 'mean'),
    avg_hour=('hour', 'mean'),
    n=('cluster', 'size'),
).reset_index()
cluster_summary['surge_pct'] = model_data_em.groupby('cluster')['surge_flag'].apply(
    lambda x: (x == "Surge").mean()).values
cluster_summary = cluster_summary.sort_values('avg_demand', ascending=False)

print(cluster_summary.round(3))

# Cluster plot
plt.figure()
sns.scatterplot(data=model_data_em, x='hour', y='ride_count', hue='cluster', alpha=0.5)
plt.title("Ride Demand Patterns by EM Cluster")
plt.xlabel("Hour of Day")
plt.ylabel("Ride Count")
plt.show()

# EM interaction model
em_interaction_model = smf.ols(
    "ride_count ~ avg_fare * cluster + avg_trip_miles + avg_trip_time + "
    "rain_flag + snow_flag + is_weekend + hour_sin + hour_cos",
    data=model_data_em
).fit()

em_interaction_results = em_interaction_model.summary2().tables[1]
em_interaction_results['conf_int_lower'] = em_interaction_model.conf_int()[0]
em_interaction_results['conf_int_upper'] = em_interaction_model.conf_int()[1]

price_interaction_effects = em_interaction_results[
    (em_interaction_results.index == "avg_fare") |
    (em_interaction_results.index.str.contains("avg_fare:cluster"))
]

print(price_interaction_effects.round(4))

coef_values = em_interaction_model.params

cluster_levels = model_data_em['cluster'].cat.categories.tolist()

cluster_price_slopes = pd.DataFrame({
    "cluster": cluster_levels,
    "estimated_fare_effect": [np.nan] * len(cluster_levels)
})

cluster_price_slopes.loc[cluster_price_slopes['cluster'] == 1, 'estimated_fare_effect'] = coef_values['avg_fare']

for cl in [c for c in cluster_levels if c != 1]:
    interaction_name = f"avg_fare:cluster[T.{cl}]"
    cluster_price_slopes.loc[cluster_price_slopes['cluster'] == cl, 'estimated_fare_effect'] = (
        coef_values['avg_fare'] + coef_values[interaction_name]
    )

print(cluster_price_slopes.round(4))

if not os.path.exists("causal_data"):
    os.makedirs("causal_data")

hourly_data.to_csv(os.path.join("causal_data", "baseline_hourly_price_data.csv"), index=False)
baseline_results.to_csv(os.path.join("causal_data", "baseline_price_model_results.csv"))
baseline_price_effect.to_csv(os.path.join("causal_data", "baseline_price_effect.csv"))

model_data_em.to_csv(os.path.join("causal_data", "em_cluster_price_data.csv"), index=False)
em_interaction_results.to_csv(os.path.join("causal_data", "em_interaction_model_results.csv"))
price_interaction_effects.to_csv(os.path.join("causal_data", "cluster_specific_price_effects.csv"))
cluster_price_slopes.to_csv(os.path.join("causal_data", "cluster_price_slopes.csv"), index=False)