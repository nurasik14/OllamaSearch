from bs4 import BeautifulSoup

import requests
from config import ( SEARCH_URL )

import truststore
truststore.inject_into_ssl()

from tqdm import tqdm

from modules import GetUrls, clean_html

query="Қылмыс"

session = requests.session()
urls, session = GetUrls(query, session)

total = len(urls)
progress_bar = None

iter = 0
while(iter < total):
  if progress_bar is None:
    progress_bar = tqdm(total=total)
    
  progress_bar.n = iter + 1
  progress_bar.refresh()
  
  url = urls[iter]
  print(f"\nFetching {url}")
  req = session.get(url, timeout=20)
  
  iter+=1
  if req.status_code == 403:
    print("No access")
    continue

  print(clean_html(req.text))