import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

model_data = pd.read_csv("analysis_data/model_data.csv")

pca_data = model_data[['avg_fare', 'avg_trip_miles', 'avg_trip_time', 'avg_speed']]

pca_scaler = StandardScaler()
pca_data_scaled = pca_scaler.fit_transform(pca_data)

pca_model = PCA()
pca_model.fit(pca_data_scaled)
print(pca_model.components_)
pca_scores_full = pca_model.transform(pca_data_scaled)
pca_scores = pd.DataFrame(pca_scores_full[:, 0:2], columns=["PC1", "PC2"])

em_data = model_data.reset_index(drop=False).rename(columns={'index': 'row_id'})
em_data['row_id'] = range(len(em_data))
em_data = em_data[['row_id', 'ride_count', 'hour', 'rain_flag', 'snow_flag', 'is_weekend']]
em_data = pd.concat([em_data.reset_index(drop=True), pca_scores.reset_index(drop=True)], axis=1)
em_data['rain_flag'] = np.where(em_data['rain_flag'] == "Rain", 1, 0)
em_data['snow_flag'] = np.where(em_data['snow_flag'] == "Snow", 1, 0)
em_data['is_weekend'] = em_data['is_weekend'].astype(str).astype(float)
em_data = em_data.dropna()

em_features = em_data.drop(columns=['row_id'])

feature_var = em_features.var()
em_features = em_features.loc[:, (~feature_var.isna()) & (feature_var > 0)]

em_scaler = StandardScaler()
em_scaled = em_scaler.fit_transform(em_features)
em_scaled = np.asarray(em_scaled)

np.random.seed(123)

best_bic = np.inf
em_model = None
for g in range(2, 5):
    gmm = GaussianMixture(n_components=g, covariance_type='full', random_state=123)
    gmm.fit(em_scaled)
    bic = gmm.bic(em_scaled)
    if bic < best_bic:
        best_bic = bic
        em_model = gmm

print(em_model)
print("BIC:", best_bic)
print("Number of components:", em_model.n_components)

model_data_em = model_data.reset_index(drop=False).rename(columns={'index': 'row_id'})
model_data_em['row_id'] = range(len(model_data_em))

em_labels = pd.DataFrame({
    'row_id': em_data['row_id'].values,
    'cluster': pd.Categorical(em_model.predict(em_scaled) + 1)
})

model_data_em = model_data_em.merge(em_labels, how='inner', on='row_id')
model_data_em = model_data_em.drop(columns=['row_id'])

print(model_data_em['cluster'].value_counts())

cluster_summary = model_data_em.groupby('cluster').agg(
    avg_demand=('ride_count', 'mean'),
    avg_hour=('hour', 'mean'),
    avg_fare=('avg_fare', 'mean'),
    avg_trip_miles=('avg_trip_miles', 'mean'),
    avg_trip_time=('avg_trip_time', 'mean'),
    avg_speed=('avg_speed', 'mean'),
    n=('cluster', 'size')
).reset_index()
cluster_summary['rain_pct'] = model_data_em.groupby('cluster')['rain_flag'].apply(lambda x: (x == "Rain").mean()).values
cluster_summary['snow_pct'] = model_data_em.groupby('cluster')['snow_flag'].apply(lambda x: (x == "Snow").mean()).values
cluster_summary['weekend_pct'] = model_data_em.groupby('cluster')['is_weekend'].apply(
    lambda x: x.astype(str).astype(float).mean()).values
cluster_summary = cluster_summary.sort_values('avg_demand', ascending=False)

print(cluster_summary)

# Plots
plt.figure()
sns.scatterplot(data=model_data_em, x='hour', y='ride_count', hue='cluster', alpha=0.6)
plt.title("EM Clustering of Ride Demand Patterns")
plt.xlabel("Hour of Day")
plt.ylabel("Ride Count")
plt.show()

# Average Demand by Hour Across Clusters
plt.figure()
sns.lineplot(data=model_data_em, x='hour', y='ride_count', hue='cluster', estimator='mean',
             errorbar=None, linewidth=1.2)
plt.title("Average Demand by Hour Across Clusters")
plt.xlabel("Hour of Day")
plt.ylabel("Average Ride Count")
plt.show()

# Demand Distribution by Cluster
plt.figure()
sns.boxplot(data=model_data_em, x='cluster', y='ride_count', hue='cluster', order=[3, 2, 4, 1])
plt.title("Demand Separation Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Ride Count")
plt.legend([], [], frameon=False)
plt.show()

model_data_em.to_csv("analysis_data/model_data_em.csv", index=False)