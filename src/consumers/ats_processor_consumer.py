import json
import logging

from src.comlib.sqs.base_consumer import BaseConsumer
from src.config.env_vars import GlobalConfig
from src.services.ats_match_inference_service import AtsMatchInferenceService
from src.common.commons_container import common_utils
from src.dtos.jobs.ats_sqs_jobs import AtsSqsJobs

logger = logging.getLogger(__name__)


class AtsProcessorConsumer(BaseConsumer):
    def __init__(self):
        super().__init__(GlobalConfig.sqs.queue_url)
        self.__ats_match_inference_service = AtsMatchInferenceService()
        self.__sqs_producer = common_utils.sqs_producer

    async def handle_message(self, message: dict) -> None:
        logger.info(f"Processing message: {message}")
        try:
            ats_sqs_jobs = AtsSqsJobs(**json.loads(message["Body"]))
        except Exception as e:
            logger.error(f"Error parsing AtsSqsJobs: {e}")
            raise e
        try:
            # await self.__ats_match_inference_service.infer_ats_match(
            #     ats_sqs_jobs.application_id, ats_sqs_jobs.job_description_id
            # )
            logger.info(
                f"Inferred ATS match: {ats_sqs_jobs.application_id}, {ats_sqs_jobs.job_description_id}"
            )
        except Exception as e:
            logger.error(f"Error inferring ATS match: {e}")
            if ats_sqs_jobs.retry_count < 3:
                await self.__sqs_producer.send_message(
                    {
                        "application_id": ats_sqs_jobs.application_id,
                        "job_description_id": ats_sqs_jobs.job_description_id,
                        "retry_count": ats_sqs_jobs.retry_count + 1,
                    },
                    delay_seconds=5 * (2**ats_sqs_jobs.retry_count),
                )
            else:
                logger.error(f"Error inferring ATS match: {e} and max retries reached")
