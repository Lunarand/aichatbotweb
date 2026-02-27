import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

# UPDATED: Pointing to the new Chat model
MODEL_PATH = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
CONTEXT_SIZE = 1024  
THREADS = 2          

print("Loading TinyLlama Chat model into RAM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_threads=THREADS,
    n_gpu_layers=0, 
    verbose=False
)
print("Model loaded successfully.")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # UPDATED: Using the official TinyLlama Chat format
    prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{request.message}</s>\n<|assistant|>\n"
    
    def generate_tokens():
        stream = llm(
            prompt,
            max_tokens=256,
            # UPDATED: Stop tokens to prevent rambling
            stop=["</s>", "<|user|>", "<|system|>"], 
            stream=True,
            echo=False
        )
        for output in stream:
            token = output["choices"][0]["text"]
            yield token.encode("utf-8")
            
    return StreamingResponse(generate_tokens(), media_type="text/event-stream")
