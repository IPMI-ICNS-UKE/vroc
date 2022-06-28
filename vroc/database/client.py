from typing import List, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import peewee

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
                orm.Image,
                orm.Run,
            )
        )

    def insert_run(
        self,
        image_id: str,
        parameters: dict,
        metric_before: float,
        metric_after: float,
        level_metrics: List[List],
    ):
        image, _ = orm.Image.get_or_create(id=image_id)
        orm.Run.create(
            image=image,
            parameters=parameters,
            metric_before=metric_before,
            metric_after=metric_after,
            level_metrics=level_metrics,
        )

    def fetch_runs(
        self, image_id: Optional[str] = None, as_dataframe: bool = False
    ) -> Union[List[dict], pd.DataFrame]:
        query = orm.Run.select()
        if image_id is not None:
            query = query.where(orm.Run.image == image_id)

        runs = list(query.dicts())

        if as_dataframe:
            runs = DatabaseClient._to_dataframe(runs)

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
    def _expand_param_dict(run: dict):
        run = run.copy()
        params = run.pop("parameters")
        for param_name, param_value in params.items():
            run[param_name] = param_value

        return run

    @staticmethod
    def _to_dataframe(runs: List[dict]) -> pd.DataFrame:
        if isinstance(runs, dict):
            runs = [runs]
        runs = [DatabaseClient._expand_param_dict(r) for r in runs]

        return pd.DataFrame.from_records(runs, index="uuid")
