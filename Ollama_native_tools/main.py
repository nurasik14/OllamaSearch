from config import ( MODEL )

# Model Related
import requests
import ollama
from ollama import chat
from modules import (
    PullTheModel, 
    GetResponse, 
    StreamResponse, 
    ModelExists,
    FindRelevantTools,
    callTheTools
)
from tools import ( available_tools_1, available_tools_2, web_fetch, web_search, search, fetch )
from config import ( SYSTEM_PROMPT, auth_header, auth_data, search_cfg, fetch_cfg )

# Exceptions
from exceptions import ModelNotFoundError, ModelNotSpecifiedError

if not MODEL:
    raise ModelNotSpecifiedError("Model not specified in .env file")

try:
    if not ModelExists(MODEL):
        raise ModelNotFoundError("Model not pulled")
    
except ModelNotFoundError:
    PullTheModel(MODEL)

if __name__ == "__main__":
    mode = int(input("Hellow, Visitor! This is my way of implementing web search & web fetch! So, before examining further choose the mode (1 - Ollama ready API, 2 - self-coded local search): "))
    print('-'*50)
    
    if mode == 1:
        requests.post("https://ollama.com/api/web_fetch", data=auth_data, headers=auth_header)
        tools = [web_search, web_fetch]
        available_tools = available_tools_1
    else:
        tools = [fetch_cfg]
        available_tools = available_tools_2
    
    
    running = True
    try:
        while(running):
            query = input("query: ")

            prompt = f"""
            USER QUESTION:{query}

            Answer the USER QUESTION using the information you collected.
            Do not summarize the sources.
            """

            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Tools mode
            called_tools = FindRelevantTools(MODEL, message, tools)
            ToolMessage = callTheTools(called_tools, available_tools)
            # print(ToolMessage)
            
            # Answering mode
            message.extend(ToolMessage)
            response_iterator = GetResponse(model=MODEL, message=message)
            print("-"*50)
            StreamResponse(response_iterator)
            
    except KeyboardInterrupt:
        message = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "USER LEFT THE CONVERSATION"}]
        response_iterator = GetResponse(model=MODEL, message=message)
        StreamResponse(response_iterator)