from typing import List

import vroc.database.models as orm
from vroc.common_types import PathLike
from vroc.logger import LoggerMixin


class DatabaseClient(LoggerMixin):
    def __init__(self, database: PathLike):
        orm.database.init(database=database)
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
