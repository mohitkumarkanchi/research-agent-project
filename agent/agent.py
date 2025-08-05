from langchain_ollama import ChatOllama
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, retriever, ollama_base_url):
        self.retriever = retriever
        self.llm = ChatOllama(base_url=ollama_base_url, model="llama3.2:latest", temperature=0)

    def answer(self, user_query):
        context = self.retriever.retrieve(user_query)
        prompt = f"Use the following context to answer the query:\n{context}\n\nQuery: {user_query}\nAnswer:"
        response = self.llm.invoke(prompt)
        logger.debug(f"LLM response: {response}")
        return response.content
