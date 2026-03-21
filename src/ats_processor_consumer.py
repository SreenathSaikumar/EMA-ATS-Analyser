import asyncio
import logging
import sys

from src.consumers.ats_processor_consumer import AtsProcessorConsumer
from src.common.commons_container import common_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ats_processor_consumer = AtsProcessorConsumer()


async def start_ats_processor_consumer():
    logger.info("Starting ATS processor consumer...")
    await ats_processor_consumer.get_messages()


if __name__ == "__main__":
    try:
        asyncio.run(start_ats_processor_consumer())
    except KeyboardInterrupt:
        logger.info("Shutting down ATS processor consumer...")
        asyncio.run(common_utils.shutdown())
    except Exception as e:
        logger.error(f"Error starting ATS processor consumer: {e}")
        sys.exit(1)
    finally:
        logger.info("Exiting...")
        sys.exit(0)
