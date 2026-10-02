# AI Document Q&A — RAG

A learning project that implements a **Retrieval-Augmented Generation (RAG)** pipeline for asking questions about a PDF document.

The system loads a PDF, splits it into text chunks, converts the chunks into vector embeddings, stores them in ChromaDB, retrieves relevant chunks for a user's question, and uses Google Gemini to generate an answer based only on the retrieved document context.

---

## Project Goal

The main goal of this project is to understand the fundamentals of **Retrieval-Augmented Generation (RAG)** and how document-based AI applications work.

The project covers:

- PDF document loading
- Text extraction
- Text chunking
- Vector embeddings
- Semantic search
- Vector databases
- Document retrieval
- Similarity/distance thresholds
- Context filtering
- LLM-based answer generation
- Source and page tracking
- Retrieval evaluation
- Answer evaluation

---

## RAG Architecture

```text
                    PDF Document
                         │
                         ▼
                ┌─────────────────┐
                │  Load PDF       │
                │  PyMuPDF        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Split Text     │
                │  into Chunks    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Generate       │
                │  Embeddings     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   ChromaDB      │
                │ Vector Database  │
                └────────┬────────┘
                         │
                         │
                User Question
                         │
                         ▼
                ┌─────────────────┐
                │ Query Embedding │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Retrieve Top-K  │
                │ Relevant Chunks │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Distance        │
                │ Threshold       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Relevant Context│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Google Gemini   │
                │ LLM             │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Answer + Sources│
                └─────────────────┘