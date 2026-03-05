import time
import pandas as pd
import requests
import logging
from urllib.parse import quote
from typing import Optional

BASE_URL = "https://www.dwds.de"

logger = logging.getLogger(__name__)

class DWDSClient:
    def __init__(self, delay: float = 0.6, timeout: int = 30):
        self.delay = delay
        self.timeout = timeout
        logger.info(f"DWDSClient initialized with {delay}s delay, {timeout}s timeout")

    def corpus_csv(self, term: str, corpus: str = "kern", limit: int = 100, sort: str = None) -> Optional[pd.DataFrame]:
        """Return concordance hits as a DataFrame

        Args:
            term: Search term
            corpus: Corpus name (default: "kern")
            limit: Maximum results to return
            sort: Sort order - "random" for random sampling, None for default (date desc)
        """
        time.sleep(self.delay)
        encoded_term = quote(term, safe='')
        url = f"{BASE_URL}/r/?q={encoded_term}&view=csv&corpus={corpus}&limit={limit}"
        if sort:
            url += f"&sort={sort}"

        try:
            logger.info(f"Fetching corpus data for '{term}' (limit={limit})")
            df = pd.read_csv(url)
            logger.info(f"✅ Successfully loaded {len(df)} corpus contexts")
            return df
            
        except pd.errors.EmptyDataError:
            logger.warning(f"No data found for term '{term}' in corpus '{corpus}'")
            return pd.DataFrame()  # Return empty DataFrame
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching corpus data for '{term}': {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error fetching corpus data for '{term}': {e}")
            return None
