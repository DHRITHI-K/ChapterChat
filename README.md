# ChapterChat — AI-Powered Study Assistant

ChapterChat is a personal AI-powered chatbot that helps students understand textbooks by answering questions directly from uploaded books.

Built using Retrieval-Augmented Generation (RAG), it ensures answers are **context-aware, accurate, and grounded in the actual content of the book**.

---

## Features

- Upload PDF or TXT books
- Ask questions and get answers from the book
- Uses semantic search (FAISS + embeddings)
- Local AI inference using LLaMA3 via Ollama (no API required)
- Shows source snippets with page references
- Generate concise book summaries
- Context-aware answers (no hallucination outside book)

---

## Tech Stack

- **Frontend/UI**: Streamlit  
- **Backend**: Python  
- **LLM**: LLaMA3 (via Ollama - local inference)  
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Vector DB**: FAISS  
- **PDF Processing**: pdfplumber  

---

## How It Works

1. User uploads a book (PDF/TXT)
2. Text is extracted and split into chunks
3. Each chunk is converted into embeddings
4. FAISS index is created for fast similarity search
5. On user query:
   - Relevant chunks are retrieved
   - Sent as context to LLaMA3
   - Model generates an answer based only on the book

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ChapterChat.git


cd ChapterChat<img width="822" height="509" alt="Screenshot 2026-03-02 at 8 20 27 PM" src="https://github.com/user-attachments/assets/e90d750f-415c-4315-b406-e85911327178" />




