# main.py
# ======================================
# Project: QueryLens (LLM File Q&A)
# Dependencies (install with pip):
# pip install gradio transformers sentence-transformers pypdf python-docx pandas
# ======================================

import gradio as gr
import pandas as pd
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

# -----------------------------
# Load models
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")   # embeddings
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")  # Q&A capable LLM

# -----------------------------
# File Loader
# -----------------------------
def load_file(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_string()
    
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    
    elif file_path.endswith(".txt"):
        return open(file_path, "r", encoding="utf-8", errors="ignore").read()
    
    else:
        return "Unsupported file format."

# -----------------------------
# Text Chunking
# -----------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# -----------------------------
# Build Vector Index
# -----------------------------
def build_index(text):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms  # normalize
    return chunks, embeddings

# -----------------------------
# Query Answering
# -----------------------------
def answer_query(query, chunks, embeddings, top_k=3):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)
    sims = np.dot(embeddings, q_vec.T).squeeze()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    context = " ".join([chunks[i] for i in top_idx])
    
    prompt = f"Answer the following question using the context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    result = qa_model(prompt, max_length=200)[0]['generated_text']
    return result

# -----------------------------
# Gradio Interface
# -----------------------------
def chat(query, file):
    if file is None or query.strip() == "":
        return "Please upload a file and enter a query."

    text = load_file(file.name)
    if text.strip() == "":
        return "Could not extract text from file."

    chunks, embeddings = build_index(text)
    return answer_query(query, chunks, embeddings)

iface = gr.Interface(
    fn=chat,
    inputs=[gr.Textbox(label="Ask something"), gr.File(label="Upload File")],
    outputs="text",
    title="QueryLens - LLM File Q&A",
    description="Upload a file (PDF, CSV, TXT, DOCX) and ask questions about its content."
)

if __name__ == "__main__":
    iface.launch()
