import os

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

load_dotenv()


class DbConfig(BaseModel):
    host: str = os.getenv("MYSQL_DB_HOST")
    port: int = os.getenv("MYSQL_DB_PORT")
    database: str = os.getenv("MYSQL_DB_DATABASE")
    username: str = os.getenv("MYSQL_DB_USER")
    password: str = os.getenv("MYSQL_DB_PASS")
    pool_size: int = int(os.getenv("MYSQL_DB_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("MYSQL_DB_MAX_OVERFLOW", "10"))

    @property
    def url(self) -> str:
        return f"mysql+aiomysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class OpenaiConfig(BaseModel):
    api_key: str = os.getenv("OPENAI_API_KEY")
    api_base: str = os.getenv("OPENAI_API_BASE")
    api_version: str = os.getenv("OPENAI_API_VERSION")
    api_model: str = os.getenv("OPENAI_API_MODEL")
    api_model_version: str = os.getenv("OPENAI_API_MODEL_VERSION")
    api_model_version: str = os.getenv("OPENAI_API_MODEL_VERSION")


class LLMConfig(BaseModel):
    model_to_use: str = os.getenv("LLM_MODEL_TO_USE", "gpt-4o-mini")

    @field_validator("model_to_use")
    def validate_model_to_use(cls, v: str) -> str:
        if v not in [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5",
        ]:
            raise ValueError("Invalid model to use")
        return v


class SqsConfig(BaseModel):
    endpoint_url: str = os.getenv("SQS_ENDPOINT_URL")
    queue_name: str = os.getenv("SQS_QUEUE_NAME")

    @property
    def queue_url(self) -> str:
        return f"{self.endpoint_url}/{self.queue_name}"


class EnvVars(BaseModel):
    db: DbConfig = DbConfig()
    openai: OpenaiConfig = OpenaiConfig()
    llm: LLMConfig = LLMConfig()
    sqs: SqsConfig = SqsConfig()


GlobalConfig = EnvVars()
