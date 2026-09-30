# Simple RAG — Local Proof of Concept

A minimal, well-structured **Retrieval-Augmented Generation** CLI
application that runs entirely on your machine using **Ollama** and
**ChromaDB**.

The goal is to prove that the RAG pipeline works correctly before
integrating it into a web application later.

---

## Project overview

You can:

- Drop PDF / DOCX / TXT / Markdown files into `documents/`.
- Run `python main.py ingest` to extract, chunk, embed, and store them.
- Run `python main.py chat` to ask questions in an interactive shell.
- Get answers **grounded only in your documents**, with source citations.
- Get a clear refusal when the question is off-topic or unsupported.

No web server, no authentication, no paid APIs — just the RAG pipeline.

---

## Architecture

```
Documents (.pdf / .docx / .txt / .md)
        │
        ▼
Document loader (PyMuPDF, python-docx)
        │
        ▼
Chunker (fixed size + overlap, boundary-aware)
        │
        ▼
Ollama embeddings (nomic-embed-text by default)
        │
        ▼
ChromaDB (cosine space, persistent on disk)
        │
        ▼
Retriever (top-K + relevance threshold)
        │
        ▼
Prompt builder (strict knowledge boundary)
        │
        ▼
Ollama LLM (qwen3:8b by default)
        │
        ▼
Answer + Sources
```

Why chunk size and overlap matter:

- **Too small** → chunks lose context, retrieval gets noisy.
- **Too large** → a single chunk covers unrelated topics, the LLM gets
  distracted, and the context window fills quickly.
- **Overlap** preserves continuity across chunk boundaries, so a
  sentence split in two can still be retrieved whole.

The defaults (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`) are a reasonable
starting point for prose. Tune them for your documents.

---

## Requirements

- Python **3.11+**
- [Ollama](https://ollama.com/download) installed and running
- Two Ollama models pulled:
  - `qwen3:8b` (or another chat model)
  - `nomic-embed-text` (or another embedding model)

---

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
```

---

## Ollama setup

1. Install Ollama: <https://ollama.com/download>
2. Start it (usually runs as a background service after install).
3. Pull the default models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

To use different models, edit `.env`:

```env
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=mxbai-embed-large
```

---

## Running

Place your documents inside `documents/`, then:

```bash
python main.py ingest     # load, chunk, embed, store
python main.py stats      # show collection + models
python main.py chat       # interactive Q&A
python main.py clear      # delete the vector collection (with prompt)
```

Add `--debug` for verbose logging:

```bash
python main.py --debug chat
```

Inside `chat`:

- `sources` — print the sources of the last answer
- `clear` — reset the in-memory conversation history
- `exit` or `quit` — leave the shell

---

## Example

```text
$ python main.py ingest
[INFO] Loading documents from /app/documents ...
[INFO] Loading document: HR_Policy.pdf
[INFO] Created 24 chunks from HR_Policy.pdf
[INFO] Loading document: Handbook.docx
[INFO] Created 43 chunks from Handbook.docx

Documents processed: 2
Chunks created: 67
Chunks stored: 67
Total chunks in collection: 67
Ingestion completed successfully.

$ python main.py chat
========================================
       Simple RAG Knowledge Agent
========================================

Type your question.
Type 'sources' to show the sources of the last answer.
Type 'exit' to quit.

You: What is the annual leave policy?

AI: According to the company policy, employees are entitled to
25 days of annual leave.

Sources:
- HR_Policy.pdf — page 4

You: What is the company's revenue?

AI: I could not find sufficient information in the knowledge base
to answer this question.

You: What is the capital of France?

AI: I can only answer questions related to the configured
knowledge base.

You: exit
```

---

## Configuration

All settings live in `.env`. See `.env.example` for the full list.

| Variable              | Default                 | Purpose                                  |
| --------------------- | ----------------------- | ---------------------------------------- |
| `OLLAMA_BASE_URL`     | `http://localhost:11434`| Ollama endpoint                          |
| `LLM_MODEL`           | `qwen3:8b`              | Chat model                               |
| `EMBEDDING_MODEL`     | `nomic-embed-text`      | Embedding model                          |
| `CHROMA_PATH`         | `./chroma_db`           | On-disk vector DB                        |
| `CHROMA_COLLECTION`   | `knowledge_base`        | Collection name                          |
| `DOCUMENTS_PATH`      | `./documents`           | Where to look for source files           |
| `CHUNK_SIZE`          | `800`                   | Characters per chunk                     |
| `CHUNK_OVERLAP`       | `150`                   | Overlap between chunks                   |
| `TOP_K`               | `5`                     | Chunks retrieved per question            |
| `RELEVANCE_THRESHOLD` | `0.70`                  | Minimum cosine similarity to accept      |
| `MAX_HISTORY_MESSAGES`| `6`                     | Previous turns kept for context          |
| `DEBUG`               | `false`                 | Verbose logging + retrieved chunk scores |

### About `RELEVANCE_THRESHOLD`

The vector store uses **cosine space**, so ChromaDB returns
`distance = 1 - cosine_similarity`. The retriever converts this back
into a **similarity in `[0, 1]`** and only keeps chunks whose
similarity is `>= RELEVANCE_THRESHOLD`.

`0.70` is a *starting point*, not a universal answer:

- Raise it (e.g. `0.80`) if you see off-topic answers.
- Lower it (e.g. `0.55`) if legitimate questions get refused.
- Different embedding models have very different similarity scales.
  Use `--debug` to see the scores and tune accordingly.

---

## RAG limitations

- Retrieval quality depends heavily on chunk size, overlap, and the
  embedding model you choose.
- Very short or very long documents may need different chunk settings.
- The similarity threshold needs tuning per corpus and per model.
- LLMs can still produce incorrect statements when the context is
  ambiguous — always check the cited sources.
- Scanned PDFs without an OCR layer will not extract any text.
- The PoC keeps conversation history in memory only; it is lost on exit.

---

## Testing

```bash
pytest -q
```

Tests mock the LLM and embeddings, so no Ollama server is required.

---

## Future architecture (not implemented)

This CLI is intentionally small so it can be lifted into a service later:

```
Vue (frontend)
     │
     ▼
FastAPI (HTTP layer)
     │
     ▼
RAG service (this code, unchanged)
     │
     ▼
Qdrant / PostgreSQL (vector + relational storage)
     │
     ▼
Ollama (LLM + embeddings)
```

The Python modules are already split so that `document_loader`,
`chunker`, `embeddings`, `vector_store`, `retriever`, `llm`, and `rag`
can be reused as-is behind a FastAPI route.