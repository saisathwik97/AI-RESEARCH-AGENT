from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent import ask_gemini
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return{
        "message": "Ai Agent running"
    }
@app.get("/ask")
def ask(question:str):
    answer=ask_gemini(question)
    return{
        "question": question,
        "answer": answer
    }   