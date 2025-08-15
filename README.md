---

# RAG Helpdesk for BigQuery AI

This project implements a **Retrieval-Augmented Generation (RAG)** helpdesk that answers technical questions about **Google BigQuery** by searching official documentation, chunking it, generating embeddings, and performing semantic vector search in BigQuery.

It supports **Together API** for embedding and LLM calls, with BigQuery as the storage and retrieval backend.

---

## Quick Start

1. **Install Python** 3.10+ and create a virtual environment.
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**: create a `.env` file in the project root and add your credentials:

   ```env
   TOGETHER_API_KEY=<YOUR_TOGETHER_API_KEY>
   GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
   ```
4. **Prepare data**:

   * Place raw documentation files in `artifacts/raw/`.
   * Generate chunks:

     ```bash
     python -m src.clean_chunk --indir ./artifacts/raw --out ./artifacts/chunks.parquet
     ```
   * Create embeddings (Together API):

     ```bash
     python -m src.embed --in ./artifacts/chunks.parquet --out ./artifacts/embeddings.parquet
     ```
   * Upload to BigQuery:

     ```bash
     python -m src.upload_bq --in ./artifacts/embeddings.parquet --limit 5000
     ```
5. **Run vector search demo in BigQuery Console**:

   ```sql
   WITH params AS (
     SELECT embedding AS q_vec
     FROM `rag_helpdesk.chunks`
     LIMIT 1
   )
   SELECT base.doc_id, base.title, base.source_uri, base.chunk_text, distance
   FROM VECTOR_SEARCH TABLE `rag_helpdesk.chunks`,
        ('SELECT q_vec FROM params'), 'embedding', 5, 'COSINE';
   ```

---

## Project Structure

```
src/           # ETL, embedding, uploader, retriever scripts
artifacts/     # raw docs, chunks parquet, embeddings parquet
README.md      # project description
requirements.txt
```

---

## How It Works (RAG Pipeline)

1. **Document Parsing** → HTML/PDF/Markdown converted to plain text.
2. **Chunking** → split into 512–1024 token segments.
3. **Embedding** → Together API generates vector embeddings.
4. **Vector Storage** → embeddings uploaded to BigQuery.
5. **Semantic Search** → VECTOR\_SEARCH finds most relevant chunks.
6. **Answer Generation** → LLM composes an answer from retrieved context.

---

## Security & Privacy

* All API keys in this package have been **redacted** or replaced with placeholders.
* Never commit real secrets to version control — use `.env` or a secrets manager.
* If you find a real API key, rotate it immediately.

---

## License

* Code: your chosen project license.
* Documentation: Google Cloud public docs (CC BY 4.0).

---
