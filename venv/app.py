import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import spacy
import time
from math import ceil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()  

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = "rag-chatbot-medical"
namespace = "ns1"

app = FastAPI()

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(name=index_name)

class UserInput(BaseModel):
    question: str

def preprocess_query(user_input):
    doc = nlp(user_input)
    key_terms = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ')]
    return ' '.join(key_terms)

@app.post("/ask-question")
async def ask_question(input_data: UserInput):
    user_input = input_data.question
    processed_input = preprocess_query(user_input)
    
    input_embedding = model.encode(processed_input).tolist()
    
    try:
        result = index.query(
            vector=input_embedding,
            top_k=3,
            include_metadata=True,
            namespace=namespace
        )
        
        if result and result['matches']:
            top_match = result['matches'][0]['metadata']['text']
            return {"response": top_match}
        else:
            raise HTTPException(status_code=404, detail="No good match found.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is live!"}

