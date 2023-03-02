from vroc.hyperopt_database.client import DatabaseClient

client = DatabaseClient("/datalake/learn2reg/param_sampling.sqlite")


# runs = client.fetch_runs("NLST_0001")
# best_runs = client.fetch_best_run("NLST_0002", k_best=3, as_dataframe=True, reduce=False)
best_runs = client.fetch_best_runs(k_best=1, as_dataframe=False)


# best_runss = best_runs.copy()
#
# best_runs.drop(
#     ["image", "metric_before", "level_metrics", "created"], axis=1, inplace=True
# )
#
# grid = sns.pairplot(
#     best_runs, hue=None, kind="kde", diag_kind="kde", plot_kws={"cmap": "plasma"}
# )
