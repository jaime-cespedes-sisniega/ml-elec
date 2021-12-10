import hashlib
import io
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import gridfs
import joblib
from ml_pipeline.base_pipeline import BasePipeline
import pymongo


class LoadingError(Exception):
    pass


class DatabaseError(Exception):
    pass


class ModelPipelineRegistryClient:

    def __init__(self, **db_kwargs):
        try:
            self.client = pymongo.MongoClient(**db_kwargs)
            self.db = self.client[db_kwargs['authSource']]
        except (pymongo.errors.PyMongoError, KeyError) as e:
            raise DatabaseError(f'Connection with database '
                                f'could not be established: {e}')
        self.fs = gridfs.GridFS(self.db)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def save_pipeline(self,
                      pipeline: BasePipeline,
                      name: str) -> None:
        """Save pipeline in database

        :param pipeline: pipeline to be saved
        :type pipeline: BasePipeline
        :param name: name of the file to be saved
        :type name: str
        :rtype: None
        """
        with TemporaryDirectory() as tmp:

            tmp_path = Path(tmp, name)
            joblib.dump(pipeline, tmp_path)

            pipeline_serialized = io.FileIO(tmp_path, 'r').read()
            _id = hashlib.md5(pipeline_serialized).hexdigest()

            try:
                _ = self.fs.put(pipeline_serialized,
                                filename=name,
                                _id=_id)
            except gridfs.errors.FileExists:
                self.logger.warning('Serialized pipeline object '
                                    'already exists. It will not be saved.')
            else:
                self.logger.info('Pipeline object was successfully saved.')

    def load_pipeline(self,
                      name: str) -> BasePipeline:
        """Load pipeline from database

        :param name: name of the file to be loaded
        :type name: str
        :return loaded pipeline
        :rtype: BasePipeline
        """
        try:
            pipeline_serialized = self.fs.get_last_version(filename=name).read()
        except gridfs.errors.NoFile as e:
            raise LoadingError(f'{name} could not be retrieved: {e}')
        pipeline = joblib.load(io.BytesIO(pipeline_serialized))
        self.logger.info('Pipeline object was successfully loaded.')
        return pipeline
