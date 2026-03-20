from abc import ABC
from logging import getLogger
from typing import Sequence

from fastapi import APIRouter
from fastapi import Depends


logger = getLogger(__name__)

class RouterBase(ABC):
    def __init__(self, prefix: str | None = None, tags: list[str] | None = None, dependencies: Sequence[Depends] | None = None):
        self.__router = APIRouter(prefix=prefix, tags=tags, dependencies=dependencies)
        self.__logger = logger

    def get_router(self):
        return self.__router