from langchain_openai import ChatOpenAI


class LLMs:
    def __init__(self):
        self.__gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.__gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.__gpt_4o_mini_azure = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0.0,
        )
        self.__gpt_4o_azure = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0.0,
        )

    def get_gpt_4o_mini(self) -> ChatOpenAI:
        return self.__gpt_4o_mini

    def get_gpt_4o(self) -> ChatOpenAI:
        return self.__gpt_4o

    def get_gpt_4o_mini_azure(self) -> ChatOpenAI:
        return self.__gpt_4o_mini_azure

    def get_gpt_4o_azure(self) -> ChatOpenAI:
        return self.__gpt_4o_azure
