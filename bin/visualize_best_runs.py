import seaborn as sns

from vroc.database.client import DatabaseClient

client = DatabaseClient("/datalake/learn2reg/merged_runs.sqlite")


runs = client.fetch_runs("NLST_0001")
best_run = client.fetch_best_run("NLST_0001")
best_runs = client.fetch_best_runs(as_dataframe=True)

best_runs.drop(
    ["image", "metric_before", "metric_after", "created"], axis=1, inplace=True
)

grid = sns.pairplot(
    best_runs, hue=None, kind="kde", diag_kind="kde", plot_kws={"cmap": "plasma"}
)
