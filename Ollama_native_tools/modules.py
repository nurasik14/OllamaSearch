import ollama
from ollama import ChatResponse
from typing import Iterator, Sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

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

def FindRelevantTools(model: str, message, tools):
    response: ChatResponse = ollama.chat(
    model,
    messages=message,
    tools=tools,
    )
    return response.message.tool_calls

def callTheTools(called_tools, available_tools):
    messages = []
    
    for tool in called_tools:
        if tool.function.name in available_tools:
            messages.append({'role': 'tool',  'tool_name': tool.function.name, 'content': str(available_tools[tool.function.name](**tool.function.arguments))})
            
    return messages

def GetResponse(model: str, message) -> Iterator[ChatResponse]:
    response = ollama.chat(
        model=model,
        messages=message,
        stream=True,
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