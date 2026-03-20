from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config.env_vars import GlobalConfig

class CommonsContainer:
    def __init__(self):
        self.db_engine = self.create_db_engine()

    @staticmethod
    def create_db_engine() -> AsyncEngine:
        return create_async_engine(
            GlobalConfig.db.url,
            pool_size=GlobalConfig.db.pool_size,
            max_overflow=GlobalConfig.db.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

common_utils = CommonsContainer()
