import pickle
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

model_data_em = pd.read_csv("analysis_data/model_data_em.csv")

# Data Types Fix

model_data_em['rain_flag'] = model_data_em['rain_flag'].astype('category')
model_data_em['snow_flag'] = model_data_em['snow_flag'].astype('category')
model_data_em['is_weekend'] = model_data_em['is_weekend'].astype('category')
model_data_em['cluster'] = model_data_em['cluster'].astype('category')

print(model_data_em.info())
print(model_data_em['cluster'].value_counts())

base_model = smf.ols(
    "ride_count ~ avg_fare + avg_trip_miles + avg_trip_time + "
    "rain_flag + snow_flag + is_weekend + hour_sin + hour_cos",
    data=model_data_em
).fit()

print(base_model.summary())

cluster_model = smf.ols(
    "ride_count ~ avg_fare + avg_trip_miles + avg_trip_time + "
    "rain_flag + snow_flag + is_weekend + hour_sin + hour_cos + cluster",
    data=model_data_em
).fit()

print(cluster_model.summary())

interaction_model = smf.ols(
    "ride_count ~ avg_fare * cluster + avg_trip_miles + avg_trip_time + "
    "rain_flag + snow_flag + is_weekend + hour_sin + hour_cos",
    data=model_data_em
).fit()

print(interaction_model.summary())

model_compare = pd.DataFrame({
    "Model": ["Base Model", "Cluster Model", "Interaction Model"],
    "AIC": [base_model.aic, cluster_model.aic, interaction_model.aic],
    "BIC": [base_model.bic, cluster_model.bic, interaction_model.bic],
    "Adjusted_R2": [base_model.rsquared_adj, cluster_model.rsquared_adj, interaction_model.rsquared_adj],
    "RMSE": [
        np.sqrt(np.mean(base_model.resid ** 2)),
        np.sqrt(np.mean(cluster_model.resid ** 2)),
        np.sqrt(np.mean(interaction_model.resid ** 2)),
    ]
})

print(model_compare.round(4))

anova_result = anova_lm(base_model, cluster_model, interaction_model)
print(anova_result)

cluster_terms = cluster_model.summary2().tables[1]
cluster_terms = cluster_terms[cluster_terms.index.str.contains("cluster")]
print(cluster_terms.round(4))

interaction_terms = interaction_model.summary2().tables[1]
interaction_terms = interaction_terms[interaction_terms.index.str.contains("avg_fare:cluster")]
print(interaction_terms.round(4))

print(model_compare.sort_values('AIC'))

with open("analysis_data/base_model.pkl", "wb") as f:
    pickle.dump(base_model, f)

with open("analysis_data/cluster_model.pkl", "wb") as f:
    pickle.dump(cluster_model, f)