from typing import List, Optional, Union

import pandas as pd

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

    def fetch_best_run(self, image_id: str) -> dict:
        query = (
            orm.Run.select()
            .where(orm.Run.image == image_id)
            .order_by(orm.Run.metric_after)
            .limit(1)
            .dicts()
            .first()
        )

        return query

    def fetch_best_runs(
        self, as_dataframe: bool = False
    ) -> Union[List[dict], pd.DataFrame]:
        best_runs = []

        for image in orm.Image.select():
            best_runs.append(self.fetch_best_run(image))

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
        runs = [DatabaseClient._expand_param_dict(r) for r in runs]

        return pd.DataFrame.from_records(runs, index="uuid")
