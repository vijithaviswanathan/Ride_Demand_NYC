import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_data_em = pd.read_csv("analysis_data/model_data_em.csv")
print(model_data_em['cluster'].value_counts())

# Validation using high demand indicator

print(pd.crosstab(model_data_em['cluster'], model_data_em['high_demand']))

cluster_high_demand_check = model_data_em.groupby('cluster').agg(
    total_obs=('cluster', 'size'),
    high_demand_pct=('high_demand', lambda x: (x == 1).mean())
).reset_index()
cluster_high_demand_check = cluster_high_demand_check.sort_values('high_demand_pct', ascending=False)

print(cluster_high_demand_check)

# Validation using surge flag

print(pd.crosstab(model_data_em['cluster'], model_data_em['surge_flag']))

cluster_surge_check = model_data_em.groupby('cluster').agg(
    total_obs=('cluster', 'size'),
    surge_pct=('surge_flag', lambda x: (x == "Surge").mean())
).reset_index()
cluster_surge_check = cluster_surge_check.sort_values('surge_pct', ascending=False)

print(cluster_surge_check)

# Visual validation
cluster_order = [3, 2, 4, 1]
cluster_high_demand_check['cluster'] = pd.Categorical(cluster_high_demand_check['cluster'], categories=cluster_order)
cluster_surge_check['cluster'] = pd.Categorical(cluster_surge_check['cluster'], categories=cluster_order)

# High demand validation plot
plt.figure()
sns.barplot(data=cluster_high_demand_check, x='cluster', y='high_demand_pct', hue='cluster', alpha=0.8)
plt.title("Validation: High Demand Percentage by Cluster")
plt.xlabel("Cluster")
plt.ylabel("High Demand Proportion")
plt.show()

# Surge validation plot
plt.figure()
sns.barplot(data=cluster_surge_check, x='cluster', y='surge_pct', hue='cluster', alpha=0.8)
plt.title("Validation: Surge Percentage by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Surge Proportion")
plt.show()