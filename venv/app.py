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

# """
# import fitz  #
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone, ServerlessSpec
# import time
# from math import ceil

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text("text")
#     return text

# pdf_paths = [
#     "../Docs/ASCO- Non-Small Cell Lung Cancer.pdf",
#     "../Docs/ASCO- Small Cell Lung Cancer.pdf",
#     "../Docs/ESMO- Early and Locally Advanced Non Small Cell Lung Cancer HP Guide.pdf",
#     "../Docs/ESMO- Malignant Pleural Mesothelioma HP Guide.pdf",
#     "../Docs/ESMO- Non Oncogene Addicted Metastatic NSCLC HP Guide.pdf",
#     "../Docs/ESMO- Non-small-cell lung cancer (NSCLC)_ An ESMO guide for patients.pdf",
#     "../Docs/ESMO- Oncogene Addicted Metastatic NSCLC HP Guide (copy).pdf",
#     "../Docs/ESMO- Small Cell Lung Cancer HP Guide.pdf",
#     "../Docs/ESMO- Thymic Epithelial Tumours HP Guide.pdf",
#     "../Docs/NCCN- Metastatic Non Small Cell Lung Cancer Patient Guide.pdf",
#     "../Docs/NCCN- Non Small Cell Lung Cancer HP Guide .pdf",
#     "../Docs/NCCN- Non Small Cell Lung Cancer Patient Guide.pdf",
#     "../Docs/NCCN- Small Cell Lung Cancer HP Guide.pdf",
#     "../Docs/NCCN- Small Cell Lung Cancer Patient Guide.pdf"
# ]

# pdf_texts = []

# for pdf_path in pdf_paths:
#     pdf_texts.append(extract_text_from_pdf(pdf_path))

# model = SentenceTransformer('all-MiniLM-L6-v2')

# pinecone_api_key = "70eaf2cd-563b-46e0-af9a-ba4667bc0198"
# pinecone_environment = "us-east-1"

# index_name = "rag-chatbot-medical"

# pc = Pinecone(api_key="70eaf2cd-563b-46e0-af9a-ba4667bc0198")

# pc.create_index(    
#     name=index_name,
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# def chunk_text(text, max_bytes):
#     chunks = []
#     current_chunk = []
#     current_size = 0
#     for sentence in text.split('. '):
#         sentence += '. '
#         sentence_size = len(sentence.encode('utf-8'))
#         if current_size + sentence_size <= max_bytes:
#             current_chunk.append(sentence)
#             current_size += sentence_size
#         else:
#             chunks.append(''.join(current_chunk))
#             current_chunk = [sentence]
#             current_size = sentence_size
#     if current_chunk:
#         chunks.append(''.join(current_chunk))
#     return chunks

# max_metadata_size = 40950

# while not pc.describe_index(index_name).status['ready']:
#     time.sleep(1)

# index = pc.Index(index_name)

# for pdf_id, pdf_text in enumerate(pdf_texts):
#     chunks = chunk_text(pdf_text, max_metadata_size)
#     chunk_embeddings = model.encode(chunks)

#     vectors = []
#     for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
#         vectors.append({
#             "id": f"pdf-{pdf_id}-chunk-{i}",
#             "values": embedding.tolist(),
#             "metadata": {'text': chunk}
#         })
    
#     index.upsert(vectors=vectors, namespace=namespace)
# """