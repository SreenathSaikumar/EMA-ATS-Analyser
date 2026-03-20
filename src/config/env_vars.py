import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class DbConfig(BaseModel):
    url: str = os.getenv("DB_URL")
    pool_size: int = os.getenv("DB_POOL_SIZE")
    max_overflow: int = os.getenv("DB_MAX_OVERFLOW")
    pool_timeout: int = os.getenv("DB_POOL_TIMEOUT")
    pool_recycle: int = os.getenv("DB_POOL_RECYCLE")
    pool_pre_ping: bool = os.getenv("DB_POOL_PRE_PING")
    pool_pre_ping_timeout: int = os.getenv("DB_POOL_PRE_PING_TIMEOUT")

class OpenaiConfig(BaseModel):
    api_key: str = os.getenv("OPENAI_API_KEY")
    api_base: str = os.getenv("OPENAI_API_BASE")
    api_version: str = os.getenv("OPENAI_API_VERSION")
    api_model: str = os.getenv("OPENAI_API_MODEL")
    api_model_version: str = os.getenv("OPENAI_API_MODEL_VERSION")
    api_model_version: str = os.getenv("OPENAI_API_MODEL_VERSION")

class RmqConfig(BaseModel):
    uri: str = os.getenv("RMQ_URI")

class EnvVars(BaseModel):
    db: DbConfig = DbConfig()
    openai: OpenaiConfig = OpenaiConfig()
    rmq: RmqConfig = RmqConfig()

GlobalConfig = EnvVars()
