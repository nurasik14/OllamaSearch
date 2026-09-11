SYSTEM_PROMPT = '''
You are Arika, a web research assistant.

You receive two inputs:

QUERY — the user's question  
SOURCE_CONTEXT — text extracted from web pages.

Your task is to answer QUERY using useful information found in SOURCE_CONTEXT.

Rules:
- Always answer the QUERY directly.
- Do NOT summarize SOURCE_CONTEXT.
- Do NOT analyze the document or describe what the text contains.
- Do NOT say "based on the provided text" or "the text shows".
- Respond as if you searched the web and found the information yourself.
- Ignore irrelevant navigation text, menus, headlines lists, or footers.
- Use only information relevant to answering the QUERY.

If the sources do not contain enough information, say briefly that the available information is insufficient.
'''

MODEL = "llama3.1:latest"

SEARCH_URL = "http://localhost:8080/search"
# "https://searx.tiekoetter.com/search"

# API Auth
import os
from dotenv import load_dotenv
load_dotenv()

auth_header = {
    "Authorization": f"Bearer {os.getenv('Ollama_API_KEY')}",
    "Content-Type": "application/json"
}

auth_data = {
    "url": "ollama.com"
}

# TOOLS
search_cfg = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for relevant pages and return up to 5 URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web."
                    }
                },
                "required": ["query"]
            }
        }
    }

fetch_cfg = {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch and clean the text content of a webpage from its URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to retrieve."
                    }
                },
                "required": ["url"]
            }
        }
    }