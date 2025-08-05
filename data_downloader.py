import requests
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

DATA_PATH = "data/papers.parquet"

class ResearchPaperDownloader:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def fetch(self, query: str, limit: int = 100):
        """
        Fetch papers metadata and abstracts.
        Fetches in batches, since Semantic Scholar API limits max per request.
        """
        all_papers = []
        batch_size = 100  # max allowed per call (adjust if API changes)
        total_fetched = 0

        while total_fetched < limit:
            fetch_count = min(batch_size, limit - total_fetched)
            params = {
                "query": query,
                "limit": fetch_count,
                "fields": "title,abstract,authors,url"
            }
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                break
            all_papers.extend(data)
            total_fetched += len(data)
            if len(data) < fetch_count:
                break  # no more data
            
            logger.info(f"Fetched {total_fetched} papers so far.")
        
        df = pd.json_normalize(all_papers)
        logger.info(f"Total papers fetched: {len(df)}")
        return df

    def save(self, df: pd.DataFrame, path=DATA_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved papers to {path}")

    def load(self, path=DATA_PATH):
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logger.info(f"Loaded papers from {path}, total {len(df)} papers.")
            return df
        else:
            logger.warning(f"No cached data found at {path}.")
            return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    query = "machine learning software engineering"
    limit = 200  # number of papers to fetch

    downloader = ResearchPaperDownloader(api_url="https://api.semanticscholar.org/graph/v1/paper/search")

    # To download fresh papers and save locally
    df = downloader.fetch(query, limit=limit)
    downloader.save(df)

    print(f"Downloaded and saved {len(df)} papers.")
