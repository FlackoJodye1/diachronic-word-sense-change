import time
import pandas as pd
import requests
import logging
from urllib.parse import quote
from typing import Dict, Any, Optional

BASE_URL = "https://www.dwds.de"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DWDSError(Exception):
    """Custom exception for DWDS API errors."""
    pass

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

    def timeseries(self, term: str, corpus: str = "kern", start: int = 1900) -> Optional[Dict[str, Any]]:
        """
        Return frequency over time as JSON (dict).
        
        Args:
            term: Search term
            corpus: Corpus name (default: "kern")
            start: Start year for analysis
            
        Returns:
            Dictionary with frequency data over time
            Returns None if error occurs
        """
        time.sleep(self.delay)
        url = f"{BASE_URL}/r/"
        
        try:
            logger.info(f"Fetching timeseries data for '{term}' from {start}")
            
            r = requests.get(url, params={
                "q": term, "view": "timeseries",
                "corpus": corpus, "format": "json",
                "date-start": start
            }, timeout=self.timeout)
            
            r.raise_for_status()
            data = r.json()
            logger.info(f"✅ Successfully loaded timeseries data for '{term}'")
            return data
            
        except requests.exceptions.HTTPError as e:
            if r.status_code == 404:
                logger.warning(f"No timeseries data found for term '{term}'")
                return {}
            else:
                logger.error(f"HTTP error {r.status_code} fetching timeseries for '{term}': {e}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching timeseries for '{term}': {e}")
            return None
            
        except ValueError as e:
            logger.error(f"JSON decode error for timeseries '{term}': {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error fetching timeseries for '{term}': {e}")
            return None
    
    def dictionary(self, term: str) -> Optional[list]:
        """
        Return dictionary entry with lexical info as JSON (list).
        
        Args:
            term: Search term
            
        Returns:
            List of dictionary entries with lemma, wortart, url
            Returns None if error occurs
        """
        time.sleep(self.delay)
        url = f"{BASE_URL}/api/wb/snippet/"
        
        try:
            logger.info(f"Fetching dictionary data for '{term}'")
            
            r = requests.get(url, params={"q": term}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                logger.warning(f"No dictionary entries found for term '{term}'")
                return []
            
            logger.info(f"✅ Successfully loaded {len(data)} dictionary entries for '{term}'")
            return data
            
        except requests.exceptions.HTTPError as e:
            if r.status_code == 404:
                logger.warning(f"No dictionary entry found for term '{term}'")
                return []
            else:
                logger.error(f"HTTP error {r.status_code} fetching dictionary for '{term}': {e}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching dictionary for '{term}': {e}")
            return None
            
        except ValueError as e:
            logger.error(f"JSON decode error for dictionary '{term}': {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error fetching dictionary for '{term}': {e}")
            return None

    def health_check(self) -> bool:
        """
        Check if DWDS API is accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            logger.info("Performing DWDS API health check...")
            r = requests.get(f"{BASE_URL}/api/wb/snippet/?q=test", timeout=10)
            healthy = r.status_code == 200
            
            if healthy:
                logger.info("✅ DWDS API is healthy")
            else:
                logger.warning(f"⚠️ DWDS API health check failed: {r.status_code}")
            
            return healthy
            
        except Exception as e:
            logger.error(f"❌ DWDS API health check failed: {e}")
            return False