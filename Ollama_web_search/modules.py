import ollama
from ollama import ChatResponse
from typing import Iterator
from tqdm import tqdm
import requests
import re
from bs4 import BeautifulSoup

from config import ( SYSTEM_PROMPT, SEARCH_URL )

def PullTheModel(model: str) -> None:
    progress_bar = None
    current_digest = None

    print(f"Loading {model}")
    for update in ollama.pull(model, stream=True):
        status = update.get("status", "")
        digest = update.get("digest")
        completed = update.get("completed", 0)
        total = update.get("total", 0)

        # If new layer starts → reset progress bar
        if digest != current_digest:
            if progress_bar:
                progress_bar.close()
            progress_bar = None
            current_digest = digest

        if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
            if progress_bar is None:
                progress_bar = tqdm(total=total, unit="B", unit_scale=True)
            progress_bar.n = completed
            progress_bar.set_description(status)
            progress_bar.refresh()
        else:
            print(status)
          
    if progress_bar:
        progress_bar.close()

    print("-" * 50)
    print("Done")
    print(f"{model} is Loaded")
    print("-" * 50)
  
def ModelExists(model: str) -> bool:
    return any(m["model"] == model for m in ollama.list()["models"])

def RemoveTheModel(model: str) -> None:
  if ModelExists(model):
      print(f"Deleting {model}")
      ollama.delete(model=model)
      print(f"{model} deleted")
  else:
      print(f"{model} not found")

def GetResponse(model: str, query: str, sources:list[str] = []) -> Iterator[ChatResponse]:
    sources_text = "\n\n".join(
        f"SOURCE {i+1}:\n{src}" for i, src in enumerate(sources)
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
    USER QUESTION:
    {query}

    INFORMATION GATHERED FROM WEB PAGES:
    {sources_text}

    Answer the USER QUESTION using the information above.
    Do not summarize the sources.
    """
            }
        ],
        stream=True
    )
    
    return response

def StreamResponse(response_iter: Iterator[ChatResponse], stream_thoughts: bool = False, stream_content:bool = True) -> dict[str, str]:
    in_thinking = False
    content = ''
    thinking = ''
    for chunk in response_iter:
        if chunk.message.thinking:
            if not in_thinking:
                in_thinking = True
                print("Thinking...\r", end="")
            
            # accumulate the partial thinking 
            thinking += chunk.message.thinking
          
            if stream_thoughts:
                print(chunk.message.thinking, end='', flush=True)
            
        elif chunk.message.content:
            if in_thinking:
                in_thinking = False
          
            # accumulate the partial content
            content += chunk.message.content
          
            if stream_content:
                print(chunk.message.content, end='', flush=True)
  
    if stream_content or stream_thoughts:
        print()
    
    return {"thinking": thinking, "content": content}

def GetUrls(query: str, session:requests.Session) -> list[str]:
    params = {
      "q": query,
      "format": "json"
    }
    
    if not session:
        results = requests.get(SEARCH_URL, params=params, timeout=20).json()['results']
    else:
        results = session.get(SEARCH_URL, params=params, timeout=20).json()['results']

      
    urls = [res['url'] for res in results]
    urls = urls[:5]
      
    return urls

def clean_html(html_content:str) -> str:
    """
    Cleans and preprocesses HTML content.
    - Removes script/style tags
    - Strips HTML tags
    - Normalizes whitespace
    - Handles HTML entities
    """
    if not isinstance(html_content, str):
        raise ValueError("Input must be a string containing HTML.")

    # Parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Get text content
    text = soup.get_text(separator=" ")

    # Remove extra whitespace and non-breaking spaces
    text = re.sub(r'\s+', ' ', text).replace(u'\xa0', ' ').strip()

    return text

def GetUrlContent(url: str, session:requests.Session) -> str:
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