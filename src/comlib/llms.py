from langchain_openai import ChatOpenAI

from src.config.env_vars import GlobalConfig


class LLMs:
    def __init__(self):
        self.__gpt_4o_mini = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=GlobalConfig.openai.api_key,
        )
        self.__gpt_4o = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=GlobalConfig.openai.api_key,
        )

    def get_gpt_4o_mini(self) -> ChatOpenAI:
        return self.__gpt_4o_mini

    def get_gpt_4o(self) -> ChatOpenAI:
        return self.__gpt_4o
