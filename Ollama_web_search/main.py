from config import ( MODEL )

# Model Related
import requests
import ollama
from ollama import chat
from modules import (
    PullTheModel, 
    GetResponse, 
    StreamResponse, 
    ModelExists
)
from tools import searching_tool


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
    running = True
    
    try:
        while(running):
            query = input()
            
            response_iterator = GetResponse(model=MODEL, query=query, sources=searching_tool(query))
            print("-"*50)
            StreamResponse(response_iterator)
            
    except KeyboardInterrupt:
        response_iterator = GetResponse("USER ENDED THE CONVERSATION", MODEL)
        StreamResponse(response_iterator)