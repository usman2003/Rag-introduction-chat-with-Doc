# Rag introduction chat with Doc

> A beginner-friendly Retrieval-Augmented Generation (RAG) demo that indexes local documents (TXT / PDF / DOCX) and answers questions grounded in those documents.  
> It returns concise answers, shows short source snippets, and carries forward the last retrieved context so follow-ups can reference recent information.

---

## What this project does (short)

I built a small RAG demo that:

1. Loads one or many local documents (`.txt`, `.pdf`, `.docx`),
2. Splits them into chunks,
3. Creates embeddings for each chunk and stores them in a local Chroma vector DB,
4. Retrieves the top-k chunks for a user question, and
5. Uses an LLM to generate a concise answer based only on the retrieved context.

It also **carries forward the last retrieval context** between questions so follow-ups can reference what was recently discussed.

---

## Why RAG & why chunking (beginner-friendly)

- **RAG** = _retrieve_ the most relevant text, then _generate_ an answer with an LLM using that text. This keeps answers grounded in your documents and reduces hallucination.
- I **chunk** documents because LLMs and vector search work better with smaller, focused pieces of text. Chunking:
  - keeps pieces within model context limits,
  - improves retrieval precision (returns relevant passages instead of whole long files),
  - speeds up search and reduces noise in results.

I use:  
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)` as a sensible default.

---

## High-level data flow

```
Files (.txt / .pdf / .docx)
    ↓ load_document(file)
    ↓ split_document(text)   ← chunking
    ↓ embed each chunk → embeddings
    ↓ store embeddings → Chroma (chroma_storage/)
    ↓ user asks question → retrieve top-k chunks
    ↓ prompt = (carried_context + new retrieved context) + question
    ↓ LLM generates concise answer (RAG)
    ↓ show answer + short source snippets (no full context printed)
```

---

## Carried-forward context (how follow-ups remember)

I keep a short-term memory so follow-up questions can build on the previous turn:

1. After answering, I save the retrieved chunks as a `new_context` string.
2. I keep a variable `previous_context` that holds the last combined context.
3. On the next question, I prepend `previous_context` to the newly retrieved chunks before building the prompt.
4. I **truncate** (`MAX_CARRIED_CONTEXT_CHARS`, e.g. `2000`) the combined context to prevent the prompt growing forever.

---

## Technologies used

- **RAG** (Retrieval-Augmented Generation) pattern
- **LangChain** (helpers for splitting and orchestration)
- **Chroma** (persistent local vector store)
- **sentence-transformers/all-MiniLM-L6-v2** (embeddings)
- **transformers** + **Qwen/Qwen2.5-3B-Instruct** (LLM generation pipeline)
- **PyPDF2** (PDF text extraction)
- **python-docx** (DOCX text extraction)
- **python-dotenv** (load Hugging Face token from `.env`)
"# Rag-introduction-chat-with-Doc" 
