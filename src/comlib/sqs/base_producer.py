import json

from logging import getLogger

from aiobotocore.session import get_session

from src.config.env_vars import GlobalConfig

logger = getLogger(__name__)


class BaseProducer:
    def __init__(self, queue_url: str):
        self.__queue_url = queue_url
        self.__session = get_session()
        self.__client = None
        self.__config = {
            "region_name": "elasticmq",
            "endpoint_url": GlobalConfig.sqs.endpoint_url,
            "aws_access_key_id": "dummy",
            "aws_secret_access_key": "dummy",
        }

    async def initialise(self) -> None:
        if not self.__client:
            self.__client = await self.__session.create_client(
                "sqs", **self.__config
            ).__aenter__()

    async def shutdown(self) -> None:
        if self.__client:
            await self.__client.__aexit__(None, None, None)
            self.__client = None

    async def send_message(self, message: dict, delay_seconds: int = 0) -> None:
        await self.initialise()
        try:
            await self.__client.send_message(
                QueueUrl=self.__queue_url,
                MessageBody=json.dumps(message),
                DelaySeconds=delay_seconds,
            )
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise e
