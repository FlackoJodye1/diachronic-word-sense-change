import io
import time
import pandas as pd
import requests
import logging
from urllib.parse import quote
from typing import Optional

BASE_URL = "https://www.dwds.de"

logger = logging.getLogger(__name__)

class DWDSClient:
    def __init__(self, delay: float = 0.6, timeout: int = 30, session_cookie: Optional[str] = None):
        self.delay = delay
        self.timeout = timeout
        self.session_cookie = session_cookie
        logger.info(f"DWDSClient initialized with {delay}s delay, {timeout}s timeout")

    def _get_csv(self, url: str) -> Optional[pd.DataFrame]:
        """Fetch a CSV URL and return as DataFrame. Attaches session cookie if set."""
        headers = {"User-Agent": "Mozilla/5.0"}
        if self.session_cookie:
            headers["Cookie"] = self.session_cookie
        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def corpus_csv(self, term: str, corpus: str = "kern", limit: int = 100, sort: str = None, paginate: bool = False) -> Optional[pd.DataFrame]:
        """Return concordance hits as a DataFrame.

        Args:
            term: Search term
            corpus: Corpus name (default: "kern")
            limit: Results per page (max 5000)
            sort: Sort order - "random" for random sampling, None for default (date_desc)
            paginate: If True, fetch all pages until exhausted. Only use with deterministic
                      sort (not "random") — random sort returns independent samples per page.
        """
        encoded_term = quote(term, safe='')
        base_url = f"{BASE_URL}/r/?q={encoded_term}&view=csv&corpus={corpus}&limit={limit}"
        if sort:
            base_url += f"&sort={sort}"

        logger.info(f"Fetching corpus data for '{term}' (corpus={corpus}, limit={limit})")

        if not paginate:
            time.sleep(self.delay)
            df = self._get_csv(base_url)
            if df is not None:
                logger.info(f"✅ Successfully loaded {len(df)} corpus contexts")
            return df

        # Paginate: loop until a page returns fewer rows than limit
        pages = []
        page = 1
        while True:
            time.sleep(self.delay)
            url = base_url + f"&p={page}"
            df = self._get_csv(url)
            if df is None or len(df) == 0:
                break
            pages.append(df)
            logger.info(f"  Page {page}: {len(df)} rows")
            if len(df) < limit:
                break
            page += 1

        if not pages:
            return pd.DataFrame()
        result = pd.concat(pages, ignore_index=True)
        logger.info(f"✅ Successfully loaded {len(result)} corpus contexts ({page} pages)")
        return result
