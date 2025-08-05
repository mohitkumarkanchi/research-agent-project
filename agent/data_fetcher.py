import requests
import logging

logger = logging.getLogger(__name__)

class ResearchPaperFetcher:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def fetch(self, query: str, limit: int = 10):
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,url"
        }
        response = requests.get(self.api_url, params=params)
        response.raise_for_status()
        data = response.json()
        logger.info(f'Fetched {len(data.get("data", []))} papers from API')
        return data.get("data", [])
