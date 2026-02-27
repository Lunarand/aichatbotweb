import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# Mount the static directory for the frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Configuration optimized for GitHub Actions (2 vCPU, ~7GB RAM)
MODEL_PATH = "tinyllama-1.1b-intermediate-step-1431k-3t.Q4_K_M.gguf"
CONTEXT_SIZE = 1024  
THREADS = 2          

print("Loading TinyLlama model into RAM...")
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
    # Raw formatting for base model. No system alignment.
    prompt = f"User: {request.message}\nBot:"
    
    def generate_tokens():
        stream = llm(
            prompt,
            max_tokens=256,
            stop=["User:", "\n\n"],
            stream=True,
            echo=False
        )
        for output in stream:
            token = output["choices"][0]["text"]
            yield token.encode("utf-8")
            
    return StreamingResponse(generate_tokens(), media_type="text/event-stream")
