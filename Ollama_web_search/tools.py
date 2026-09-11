import requests

from modules import ( GetUrls, GetUrlContent )

def searching_tool( query:str ) -> list[str]:
    try:
        session = requests.Session()
        searching_results = GetUrls(query, session)
        sources = [GetUrlContent(url, session) for url in searching_results]
        return sources
    except:
        return []
    