from abc import ABC
from logging import getLogger

logger = getLogger(__name__)

class BaseController(ABC):
    def __init__(self):
        self.__logger = logger