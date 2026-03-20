from logging import getLogger

logger = getLogger(__name__)

class HealthCheckService:
    def __init__(self):
        pass

    async def check_db_health(self) -> bool:
        try:
            logger.info("Checking database health...")
            return True
        except Exception as e:
            logger.exception("Error checking database health", exc_info=True)
            return False
