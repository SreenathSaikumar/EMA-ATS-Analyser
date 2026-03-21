import asyncio
import logging

import dotenv

from aiobotocore.session import get_session

from src.config.env_vars import GlobalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


async def create_sqs_queues():
    logger.info("Creating SQS queues...")
    session = get_session()
    config = {
        "region_name": "elasticmq",
        "endpoint_url": GlobalConfig.sqs.endpoint_url,
        "aws_access_key_id": "dummy",
        "aws_secret_access_key": "dummy",
    }
    try:
        async with session.create_client("sqs", **config) as client:
            logger.info(f"Creating SQS queue: {GlobalConfig.sqs.queue_url}")
            await client.create_queue(
                QueueName=GlobalConfig.sqs.queue_name,
                Attributes={
                    "DelaySeconds": "0",
                    "VisibilityTimeout": "300",
                    "MessageRetentionPeriod": "86400",
                },
            )
        logger.info("SQS queue created successfully")
    except Exception as e:
        logger.error(f"Error creating SQS queue: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(create_sqs_queues())
