from ollama import web_fetch, web_search
from config import SEARCH_URL
from modules import clean_html
import requests

def search(query: str, session:requests.Session=None) -> list[str]:
    params = {
      "q": query,
      "format": "json"
    }
    
    if session is None:
        results = requests.get(SEARCH_URL, params=params, timeout=20).json()['results']
    else:
        results = session.get(SEARCH_URL, params=params, timeout=20).json()['results']

      
    urls = [res.get('url') for res in results if res.get('url')]
    urls = urls[:5]
      
    return urls

def fetch(url: str, session:requests.Session=None) -> str:
    print(f"Fetching {url}")

    try:
        if session is None:
            req = requests.get(url, timeout=(10, 20))
        else:
            req = session.get(url, timeout=(10, 20))

        if req.status_code == 403:
            print(f"No access to the page: {req.status_code}")
            return ""

        req.raise_for_status()
        return "\n" + clean_html(req.text)

    except requests.exceptions.Timeout:
        print(f"Timeout trying to access the page {url}")
        return ""

    except requests.exceptions.ConnectionError as e:
        print(f"Connection error while accessing {url}: {e}")
        return ""

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return ""

available_tools_1 = {'web_search': web_search, 'web_fetch': web_fetch}
available_tools_2 = {'search': search, 'fetch': fetch}