from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .settings import OPENAI_API_KEY

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

def get_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
