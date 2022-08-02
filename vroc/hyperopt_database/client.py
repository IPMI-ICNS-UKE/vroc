from typing import List, Optional, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from peewee import prefetch

import vroc.database.models as orm
from vroc.common_types import PathLike
from vroc.logger import LoggerMixin


class DatabaseClient(LoggerMixin):
    def __init__(self, database: PathLike):
        self.database = orm.database
        self.database.init(database=database)
        self._create_tables()

    def _create_tables(self):
        orm.database.create_tables(
            (
                orm.Modality,
                orm.Anatomy,
                orm.Dataset,
                orm.Metric,
                orm.Image,
                orm.ImagePairFeature,
                orm.Run,
                orm.RunMetrics,
            )
        )

    def insert_metric(self, name: str, lower_is_better: bool = True) -> orm.Metric:
        return orm.Metric.create(name=name, lower_is_better=lower_is_better)

    def fetch_metric(self, name: str) -> orm.Metric:
        return orm.Metric.get(name=name)

    def get_or_create_image(
        self, image_name: str, modality: str, anatomy: str, dataset: str
    ) -> orm.Image:
        modality, _ = orm.Modality.get_or_create(name=modality.upper())
        anatomy, _ = orm.Anatomy.get_or_create(name=anatomy.upper())
        dataset, _ = orm.Dataset.get_or_create(name=dataset.upper())

        return orm.Image.get_or_create(
            name=image_name, modality=modality, anatomy=anatomy, dataset=dataset
        )[0]

    def fetch_image(self, uuid: UUID) -> orm.Image:
        return orm.Image.get(uuid=uuid)

    def insert_run(
        self, moving_image: orm.Image, fixed_image: orm.Image, parameters: dict
    ) -> orm.Run:
        return orm.Run.create(
            moving_image=moving_image, fixed_image=fixed_image, parameters=parameters
        )

    def insert_run_metric(
        self, run: orm.Run, metric: orm.Metric, value_before: float, value_after: float
    ):
        return orm.RunMetrics.create(
            run=run, metric=metric, value_before=value_before, value_after=value_after
        )

    def fetch_runs(
        self,
        moving_image: orm.Image,
        fixed_image: orm.Image,
        as_dataframe: bool = False,
    ) -> Union[List[dict], pd.DataFrame]:
        runs = orm.Run.select().where(
            (orm.Run.moving_image == moving_image)
            & (orm.Run.fixed_image == fixed_image)
        )

        runs = prefetch(runs, orm.RunMetrics)

        def to_dict(run: orm.Run):
            data = run.__data__.copy()
            data["run_metrics"] = [
                {
                    "metric": run_metric.metric.name,
                    "value_before": run_metric.value_before,
                    "value_after": run_metric.value_after,
                }
                for run_metric in run.run_metrics
            ]

            return data

        runs = [to_dict(run) for run in runs]

        if as_dataframe:
            runs = DatabaseClient._runs_to_dataframe(runs)

        return runs

    def fetch_best_run(
        self,
        image_id: str,
        k_best: int = 1,
        reduce: bool = True,
        as_dataframe: bool = False,
    ) -> Union[dict, List[dict], pd.DataFrame]:
        query = (
            orm.Run.select()
            .where(orm.Run.image == image_id)
            .order_by(orm.Run.metric_after.asc())
            .limit(k_best)
            .dicts()
            .first(n=k_best)
        )
        if k_best > 1 and reduce:
            query = {
                "uuid": uuid4(),
                "image": query[0]["image"],
                "parameters": self._reduce_parameters([q["parameters"] for q in query]),
                "metric_before": float(np.mean([q["metric_before"] for q in query])),
                "metric_after": float(np.mean([q["metric_after"] for q in query])),
                "level_metrics": None,
                "created": None,
            }

        if as_dataframe:
            query = DatabaseClient._to_dataframe(query)

        return query

    def _reduce_parameters(self, parameters: List[dict]):
        reduced = {}
        for params in parameters:
            for param_name, param_value in params.items():
                try:
                    reduced[param_name].append(param_value)
                except KeyError:
                    reduced[param_name] = [param_value]

        # reduce
        for param_name, param_values in reduced.items():
            reduced[param_name] = np.mean(param_values)

        return reduced

    def fetch_best_runs(
        self, k_best: int = 1, as_dataframe: bool = False
    ) -> Union[List[dict], pd.DataFrame]:
        best_runs = []

        for image in orm.Image.select():
            best_runs.append(self.fetch_best_run(image, k_best=k_best))

        if as_dataframe:
            best_runs = DatabaseClient._to_dataframe(best_runs)

        return best_runs

    @staticmethod
    def _expand_dict(run: dict, key: str, prefix: str = ""):
        run = run.copy()
        params = run.pop(key)
        for param_name, param_value in params.items():
            run[prefix + param_name] = param_value

        return run

    @staticmethod
    def _runs_to_dataframe(runs: List[dict]) -> pd.DataFrame:
        if isinstance(runs, dict):
            runs = [runs]
        for i in range(len(runs)):
            run = runs[i]

            # include parameters nested dict in top level
            parameters = run.pop("parameters")
            for param_name, param_value in parameters.items():
                run[param_name] = param_value

            # include run_metrics nested list of dicts in top level
            run_metrics = run.pop("run_metrics")
            for run_metric in run_metrics:
                metric_name = run_metric["metric"].lower()
                run[f"{metric_name}_before"] = run_metric["value_before"]
                run[f"{metric_name}_after"] = run_metric["value_after"]

            # put modified run dict back into list
            runs[i] = run

        return pd.DataFrame.from_records(runs, index="uuid")


if __name__ == "__main__":

    client = DatabaseClient("/home/fmadesta/research/varreg_on_crack/test_db.sqlite")

    tre_metric = client.fetch_metric(name="TRE")
    moving_image = client.get_or_create_image(
        image_name="moving_image", modality="CT", anatomy="lung", dataset="NLST"
    )
    fixed_image = client.get_or_create_image(
        image_name="fixed_image", modality="CT", anatomy="lung", dataset="NLST"
    )

    # run = client.insert_run(
    #     moving_image=moving_image,
    #     fixed_image=fixed_image,
    #     parameters={'some': 'params'},
    #
    # )
    # run_metric = client.insert_run_metric(run=run, metric=tre_metric, value_before=1337, value_after=42)
    runs = client.fetch_runs(moving_image, fixed_image, as_dataframe=True)
