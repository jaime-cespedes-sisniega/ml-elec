from pathlib import Path
from typing import Optional

from pydantic import (BaseSettings,
                      validator)


class FileExtensionError(Exception):
    """FileExtensionError Exception"""

    pass


class NonExistingTestFileError(Exception):
    """NonExistingTestFileError Exception"""

    pass


class DataSettings(BaseSettings):
    """Database settings class

    Set database variables to be used
    """

    PATH: str


class PipelineSettings(BaseSettings):
    """Pipeline settings class

    Set pipeline variables to be used
    """

    TRAIN_FILE_NAME: str
    TEST_FILE_NAME: Optional[str]
    TARGET_NAME: str
    RANDOM_STATE: int
    OPTIMIZATION_TRIALS: int
    TEST: bool

    @validator('TEST')
    def _test_existing_file(cls, value, values):  # noqa: N805
        if value and not values['TEST_FILE_NAME']:
            raise NonExistingTestFileError('TEST_FILE_NAME must be set '
                                           'if TEST equals to True')
        return values

    @validator('TRAIN_FILE_NAME', 'TEST_FILE_NAME')
    def _file_name_extension(cls, value, **kwargs):  # noqa: N805
        if value and not value.endswith('.csv'):
            raise FileExtensionError(f'{kwargs["field"].name} '
                                     f'must be a csv file')
        return value

    @validator('OPTIMIZATION_TRIALS')
    def _trials_value(cls, value):  # noqa: N805
        if value < 1:
            raise ValueError('OPTIMIZATION_CV value must be greater than 0')
        return value


class ModelRegistrySettings(BaseSettings):
    """Model registry settings class

    Set model registry variables to be used
    """

    MLFLOW_HOST: str
    MLFLOW_PORT: int
    MLFLOW_USERNAME: str
    MLFLOW_PASSWORD: str
    MLFLOW_SERVER_CERT_PATH: str
    MODEL_NAME: str
    MINIO_HOST: str
    MINIO_PORT: int
    MINIO_USERNAME: str
    MINIO_PASSWORD: str


class DriftSettings(BaseSettings):
    """Drift settings class

    Set drift variables to be used
    """

    ERT: int
    WINDOW_SIZE: int
    N_BOOTSTRAP: int
    SAMPLE_RATIO: float

    @validator('SAMPLE_RATIO')
    def _sample_ratio_range(cls, value):  # noqa: N805
        if not 0.0 < value <= 1.0:
            raise ValueError('SAMPLE_RATIO value must be in (0, 1]')
        return value


class Settings(BaseSettings):
    """Settings class

    Set variables to be used
    """

    DATA: DataSettings
    PIPELINE: PipelineSettings
    MODEL_REGISTRY: ModelRegistrySettings
    DRIFT: DriftSettings

    class Config:
        """Config class

        Set env file to read
        """

        env_file = Path(__file__).parent.parent / '.env'
        env_file_encoding = 'utf-8'
        env_nested_delimiter = '__'
        case_sensitive = True


settings = Settings()
