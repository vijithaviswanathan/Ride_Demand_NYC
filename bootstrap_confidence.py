import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

np.random.seed(123)

causal_data_path = os.path.join("causal_data", "baseline_hourly_price_data.csv")

if os.path.exists(causal_data_path):
    hourly_data = pd.read_csv(causal_data_path)
else:
    rides_weather = pd.read_csv(os.path.join("data", "rides_weather_merged.csv"))

    rides_weather['pickup_datetime'] = pd.to_datetime(rides_weather['pickup_datetime'])
    rides_weather['date'] = rides_weather['pickup_datetime'].dt.date
    rides_weather['hour'] = rides_weather['pickup_datetime'].dt.hour
    rides_weather['day_of_week'] = rides_weather['pickup_datetime'].dt.strftime('%a')
    rides_weather['is_weekend'] = np.where(rides_weather['day_of_week'].isin(["Sat", "Sun"]), 1, 0)
    rides_weather['rain_flag'] = np.where(rides_weather['PRCP'] > 0, 1, 0)
    rides_weather['snow_flag'] = np.where(rides_weather['SNOW'] > 0, 1, 0)

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
    hourly_data = hourly_data.dropna(subset=['ride_count', 'avg_fare', 'hour', 'is_weekend',
                                              'TAVG', 'rain_flag', 'snow_flag'])

hourly_data = hourly_data.dropna(subset=['ride_count', 'avg_fare', 'hour', 'is_weekend', 'TAVG',
                                          'rain_flag', 'snow_flag', 'avg_trip_miles', 'avg_trip_time'])

hourly_data_summary = pd.DataFrame({
    "total_hours": [len(hourly_data)],
    "avg_hourly_demand": [hourly_data['ride_count'].mean()],
    "mean_fare": [hourly_data['avg_fare'].mean()],
    "sd_fare": [hourly_data['avg_fare'].std()],
    "min_fare": [hourly_data['avg_fare'].min()],
    "max_fare": [hourly_data['avg_fare'].max()],
})

print(hourly_data_summary.round(3))

original_model = smf.ols(
    "ride_count ~ avg_fare + hour + is_weekend + TAVG + rain_flag + snow_flag + "
    "avg_trip_miles + avg_trip_time",
    data=hourly_data
).fit()

original_results = original_model.summary2().tables[1]
original_results['conf_int_lower'] = original_model.conf_int()[0]
original_results['conf_int_upper'] = original_model.conf_int()[1]
original_price_effect = original_results.loc[['avg_fare']]

print(original_price_effect.round(4))

B = 1000
n = len(hourly_data)

bootstrap_records = []
for i in range(1, B + 1):
    boot_sample = hourly_data.sample(n=n, replace=True)

    boot_model = smf.ols(
        "ride_count ~ avg_fare + hour + is_weekend + TAVG + rain_flag + snow_flag + "
        "avg_trip_miles + avg_trip_time",
        data=boot_sample
    ).fit()

    bootstrap_records.append({
        "bootstrap_sample": i,
        "price_effect": boot_model.params.get("avg_fare", np.nan)
    })

bootstrap_price_effects = pd.DataFrame(bootstrap_records)

bootstrap_summary = pd.DataFrame({
    "bootstrap_mean": [bootstrap_price_effects['price_effect'].mean()],
    "bootstrap_sd": [bootstrap_price_effects['price_effect'].std()],
    "ci_lower": [bootstrap_price_effects['price_effect'].quantile(0.025)],
    "ci_upper": [bootstrap_price_effects['price_effect'].quantile(0.975)],
    "percent_negative": [(bootstrap_price_effects['price_effect'] < 0).mean() * 100],
})

print(bootstrap_summary.round(4))

# Distribution Plot

plt.figure()
sns.histplot(bootstrap_price_effects['price_effect'], bins=30, alpha=0.7)
plt.axvline(x=bootstrap_summary['ci_lower'].iloc[0], linestyle='--')
plt.axvline(x=bootstrap_summary['ci_upper'].iloc[0], linestyle='--')
plt.axvline(x=0, linestyle='-')
plt.title("Bootstrap Distribution of Estimated Price Effect")
plt.suptitle("Dashed lines show the 95% bootstrap confidence interval; solid line marks zero")
plt.xlabel("Estimated Effect of Average Fare on Hourly Ride Demand")
plt.ylabel("Bootstrap Frequency")
plt.show()

em_path = os.path.join("causal_data", "em_cluster_price_data.csv")

bootstrap_cluster_effects = None
cluster_bootstrap_summary = None

if os.path.exists(em_path):
    model_data_em = pd.read_csv(em_path)
    model_data_em['rain_flag'] = model_data_em['rain_flag'].astype('category')
    model_data_em['snow_flag'] = model_data_em['snow_flag'].astype('category')
    model_data_em['is_weekend'] = model_data_em['is_weekend'].astype('category')
    model_data_em['cluster'] = model_data_em['cluster'].astype('category')
    model_data_em = model_data_em.dropna(subset=['ride_count', 'avg_fare', 'cluster', 'avg_trip_miles',
                                                  'avg_trip_time', 'rain_flag', 'snow_flag', 'is_weekend',
                                                  'hour_sin', 'hour_cos'])

    B_em = 1000
    n_em = len(model_data_em)
    clusters = model_data_em['cluster'].cat.categories.tolist()

    cluster_records = []
    for i in range(1, B_em + 1):
        boot_sample = model_data_em.sample(n=n_em, replace=True)

        boot_model = smf.ols(
            "ride_count ~ avg_fare * cluster + avg_trip_miles + avg_trip_time + "
            "rain_flag + snow_flag + is_weekend + hour_sin + hour_cos",
            data=boot_sample
        ).fit()

        coef_values = boot_model.params

        for cl in clusters:
            if cl == clusters[0]:
                slope = coef_values.get("avg_fare", np.nan)
            else:
                interaction_name = f"avg_fare:cluster[T.{cl}]"
                slope = coef_values.get("avg_fare", np.nan) + coef_values.get(interaction_name, np.nan)

            cluster_records.append({
                "bootstrap_sample": i,
                "cluster": cl,
                "price_effect": slope
            })

    bootstrap_cluster_effects = pd.DataFrame(cluster_records)

    cluster_bootstrap_summary = bootstrap_cluster_effects.groupby('cluster').agg(
        bootstrap_mean=('price_effect', 'mean'),
        bootstrap_sd=('price_effect', 'std'),
        ci_lower=('price_effect', lambda x: x.quantile(0.025)),
        ci_upper=('price_effect', lambda x: x.quantile(0.975)),
        percent_negative=('price_effect', lambda x: (x < 0).mean() * 100),
    ).reset_index()

    print(cluster_bootstrap_summary.round(4))
else:
    print("EM cluster data not found. Skipping optional cluster bootstrap section.")

if bootstrap_cluster_effects is not None:
    g = sns.FacetGrid(bootstrap_cluster_effects, col='cluster', sharey=False)
    g.map(sns.histplot, 'price_effect', bins=30, alpha=0.7)
    for ax in g.axes.flat:
        ax.axvline(x=0, linestyle='-')
    g.figure.suptitle("Bootstrap Price Effect Distribution by EM Cluster")
    g.set_axis_labels("Estimated Fare Effect", "Bootstrap Frequency")
    plt.show()

if not os.path.exists("bootstrap_data"):
    os.makedirs("bootstrap_data")

bootstrap_price_effects.to_csv(os.path.join("bootstrap_data", "bootstrap_price_effects.csv"), index=False)
bootstrap_summary.to_csv(os.path.join("bootstrap_data", "bootstrap_price_effect_summary.csv"), index=False)

if bootstrap_cluster_effects is not None:
    bootstrap_cluster_effects.to_csv(os.path.join("bootstrap_data", "bootstrap_cluster_price_effects.csv"), index=False)
    cluster_bootstrap_summary.to_csv(os.path.join("bootstrap_data", "bootstrap_cluster_price_effect_summary.csv"), index=False)