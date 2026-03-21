import asyncio

from logging import getLogger
from abc import ABC, abstractmethod

from aiobotocore.session import get_session
from aiobotocore.client import BaseClient

from src.config.env_vars import GlobalConfig

logger = getLogger(__name__)


class BaseConsumer(ABC):
    def __init__(self, queue_url: str):
        self.__queue_url = queue_url

    async def get_messages(self) -> list[dict]:
        session = get_session()
        config = {
            "region_name": "elasticmq",
            "endpoint_url": GlobalConfig.sqs.endpoint_url,
            "aws_access_key_id": "dummy",
            "aws_secret_access_key": "dummy",
        }
        while True:
            async with session.create_client("sqs", **config) as client:
                try:
                    response = await client.receive_message(
                        QueueUrl=self.__queue_url,
                        MaxNumberOfMessages=10,
                        WaitTimeSeconds=5,
                    )
                    if "Messages" in response:
                        await asyncio.gather(
                            *[
                                self.__process_message(client, message)
                                for message in response["Messages"]
                            ]
                        )
                except Exception as e:
                    logger.error(f"Error processing messages: {e}")

    async def __process_message(self, client: BaseClient, message: dict) -> None:
        try:
            await self.handle_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
        finally:
            try:
                await self.__delete_message(client, self.__queue_url, message)
            except Exception as ex:
                logger.error(f"Error deleting message: {ex}")

    @abstractmethod
    async def handle_message(self, message: dict) -> None:
        pass

    @staticmethod
    async def __delete_message(
        client: BaseClient, queue_url: str, message: dict
    ) -> None:
        await client.delete_message(
            QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
        )
