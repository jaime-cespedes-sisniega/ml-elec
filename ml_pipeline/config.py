from pathlib import Path

from pydantic import (BaseSettings,
                      validator)


class ExtensionError(Exception):
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
    TEST_FILE_NAME: str
    TARGET_NAME: str
    RANDOM_STATE: int
    OPTIMIZATION_TRIALS: int
    OPTIMIZATION_CV: int
    TEST: bool

    @validator('TRAIN_FILE_NAME')
    def train_file_name_extension(cls, value):
        if not value.endswith('.csv'):
            raise ExtensionError('TRAIN_FILE_NAME must be a csv file')
        return value


class ModelRegistrySettings(BaseSettings):
    """Model registry settings class

    Set model registry variables to be used
    """

    MLFLOW_HOST: str
    MLFLOW_PORT: int
    MLFLOW_USERNAME: str
    MLFLOW_PASSWORD: str
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
    def sample_ratio_range(cls, value):
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