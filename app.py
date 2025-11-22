import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# -----------------------
# OLLAMA CONFIG (local LLM)
# -----------------------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # make sure you've run: `ollama pull llama3`


# -----------------------
# Helper: call local LLM via Ollama
# -----------------------
def query_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "Sorry, I couldn’t generate a response.")
    except Exception as e:
        return f"Error talking to local model: {e}"


# -----------------------
# PDF / Text processing
# -----------------------
def extract_chunks_from_pdf(uploaded_file, chunk_size=200):
    """
    Extract text from PDF and return list of chunks with page numbers:
    [ {"page": 1, "text": "..."}, ... ]
    """
    chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue
            words = page_text.split()
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i + chunk_size])
                chunks.append({"page": page_num, "text": chunk_text})
    return chunks


def extract_chunks_from_txt(uploaded_file, chunk_size=200):
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        # page is "N/A" for txt files
        chunks.append({"page": "N/A", "text": chunk_text})
    return chunks


# Load embed model only once
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_faiss_index(chunks):
    """
    chunks: list of {"page": int/str, "text": str}
    """
    texts = [c["text"] for c in chunks]
    embeddings = EMBED_MODEL.encode(texts)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype="float32"))

    return index, embeddings, chunks


def search_index(user_query, index, chunks, embeddings, top_k=3):
    q_emb = EMBED_MODEL.encode([user_query])
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)
    return [chunks[i] for i in I[0]]


# -----------------------
# Streamlit UI
# -----------------------
st.title("📚 ChapterChat - Your Study Buddy")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None  # list of {"page", "text"}
    st.session_state.embeddings = None

# 📂 Upload Book
uploaded_file = st.file_uploader("Upload your book (PDF or TXT)", type=["pdf", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        chunks = extract_chunks_from_pdf(uploaded_file)
    else:
        chunks = extract_chunks_from_txt(uploaded_file)

    index, embeddings, chunks = build_faiss_index(chunks)

    st.session_state.index = index
    st.session_state.embeddings = embeddings
    st.session_state.chunks = chunks

    st.success("✅ Book uploaded and indexed!")

# 📝 Book summary button
if st.session_state.chunks:
    if st.button("📝 Summarize this book for me"):
        # Use first N chunks to avoid overloading the model
        sample_chunks = st.session_state.chunks[:25]  # adjust if needed
        context = " ".join(c["text"] for c in sample_chunks)

        summary_prompt = (
            "You are a helpful study assistant. "
            "Read the following textbook content and create a short, clear summary "
            "in bullet points for a student:\n\n"
            f"{context}\n\nSummary (use bullet points):"
        )

        with st.spinner("Generating summary..."):
            summary = query_ollama(summary_prompt)

        st.markdown("### 📄 Book Summary")
        st.markdown(summary)

# 💬 Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 💬 Chat logic
if prompt := st.chat_input("Ask me anything about your book..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index is not None:
        relevant_chunks = search_index(
            prompt,
            st.session_state.index,
            st.session_state.chunks,
            st.session_state.embeddings
        )

        context = " ".join(c["text"] for c in relevant_chunks)

        final_prompt = (
            "You are a helpful study assistant. "
            "Use ONLY the following book context to answer the question. "
            "If the answer is not in the context, say you are not sure.\n\n"
            f"Book context:\n{context}\n\n"
            f"Question: {prompt}\n\nAnswer:"
        )

        bot_reply = query_ollama(final_prompt)
    else:
        bot_reply = query_ollama(prompt)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

        # 📖 Show sources under the answer
        if st.session_state.index is not None:
            with st.expander("📖 Sources from your book"):
                for i, chunk in enumerate(relevant_chunks, start=1):
                    page = chunk.get("page", "N/A")
                    snippet = chunk["text"]
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "..."
                    st.markdown(f"**Source {i} (Page {page}):**\n\n{snippet}")

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})