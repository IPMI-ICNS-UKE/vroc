import uuid

from peewee import (
    BlobField,
    CharField,
    FloatField,
    ForeignKeyField,
    Model,
    SqliteDatabase,
    UUIDField,
)

from vroc.hyperopt_database.fields import JSONField

database = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = database


class Modality(BaseModel):
    name = CharField(primary_key=True)


class Anatomy(BaseModel):
    name = CharField(primary_key=True)


class Dataset(BaseModel):
    name = CharField(primary_key=True)


class Image(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    name = CharField(max_length=255)
    modality = ForeignKeyField(Modality, backref="modalities", on_delete="CASCADE")
    anatomy = ForeignKeyField(Anatomy, backref="anatomies", on_delete="CASCADE")
    dataset = ForeignKeyField(Dataset, backref="datasets", on_delete="CASCADE")


class BestParameters(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    moving_image = ForeignKeyField(Image, backref="moving_images", on_delete="CASCADE")
    fixed_image = ForeignKeyField(Image, backref="fixed_images", on_delete="CASCADE")
    parameters = JSONField()

    # performance
    metric_before = FloatField()
    metric_after = FloatField()


class ImagePairFeatures(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    moving_image = ForeignKeyField(Image, backref="moving_images", on_delete="CASCADE")
    fixed_image = ForeignKeyField(Image, backref="fixed_images", on_delete="CASCADE")
    features = BlobField()
