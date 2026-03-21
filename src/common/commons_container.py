from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config.env_vars import GlobalConfig
from src.comlib.sqs.base_producer import BaseProducer


class CommonsContainer:
    def __init__(self):
        self.db_engine = self.create_db_engine()
        self.sqs_producer = self.create_sqs_producer()

    @staticmethod
    def create_db_engine() -> AsyncEngine:
        return create_async_engine(
            GlobalConfig.db.url,
            pool_size=GlobalConfig.db.pool_size,
            max_overflow=GlobalConfig.db.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    @staticmethod
    def create_sqs_producer() -> BaseProducer:
        return BaseProducer(GlobalConfig.sqs.queue_url)

    async def shutdown(self) -> None:
        await self.db_engine.dispose()
        await self.sqs_producer.shutdown()


common_utils = CommonsContainer()
