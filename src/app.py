import logging

from fastapi import FastAPI

from src.containers.api_container import ApiContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ATS Analyser", description="ATS Analyser")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    api_container = ApiContainer()
    routers = await api_container.get_routers()
    for router in routers:
        app.include_router(router)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
