
-- === Create VECTOR INDEX (optional but recommended for scale) ===
-- Docs: https://cloud.google.com/bigquery/docs/vector-search-intro
-- Use IVF (TreeAH also available). For IVF, tune num_lists based on table size.

-- Replace YOUR_PROJECT_ID and dataset/table if needed
CREATE OR REPLACE VECTOR INDEX `YOUR_PROJECT_ID.rag_helpdesk.idx_chunks_ivf`
ON `YOUR_PROJECT_ID.rag_helpdesk.chunks` (embedding)
OPTIONS(
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists":500}'
);

-- Check index build status:
-- SELECT * FROM `YOUR_PROJECT_ID`.`rag_helpdesk`.INFORMATION_SCHEMA.VECTOR_INDEXES;

-- === Sample VECTOR_SEARCH() ===
-- Query vector can come from local embedding (client) or from BigQuery ML.GENERATE_EMBEDDING:
-- Example using ML.GENERATE_EMBEDDING with Gemini (requires setup & billing):
-- SELECT ML.GENERATE_EMBEDDING(MODEL `region-us`.GENERATIVE_AI.EMBED_TEXT, STRUCT('your prompt here' AS content, TRUE AS flatten_json_output));

DECLARE top_k INT64 DEFAULT 5;
DECLARE q_vec ARRAY<FLOAT64> DEFAULT [0.0, 0.0, 0.0]; -- placeholder; pass real vector from client via parameter

-- Brute-force search (no index)
SELECT doc_id, chunk_id, title, source_uri, chunk_text, distance
FROM VECTOR_SEARCH(
  TABLE `YOUR_PROJECT_ID.rag_helpdesk.chunks`,
  'embedding',
  q_vec,
  top_k,
  OPTIONS (distance_type = 'COSINE')
);

-- Indexed search (after index ready)
SELECT doc_id, chunk_id, title, source_uri, chunk_text, distance
FROM VECTOR_SEARCH(
  TABLE `YOUR_PROJECT_ID.rag_helpdesk.chunks`,
  'embedding',
  q_vec,
  top_k,
  OPTIONS (distance_type = 'COSINE', use_index = TRUE, fraction_lists_to_search = 0.05)
);

-- Filtering example (source_uri or title)
-- Add WHERE clause after wrapping VECTOR_SEARCH in subquery:
-- SELECT * FROM (
--   SELECT doc_id, chunk_id, title, source_uri, chunk_text, distance
--   FROM VECTOR_SEARCH(TABLE `YOUR_PROJECT_ID.rag_helpdesk.chunks`, 'embedding', q_vec, 20, OPTIONS(distance_type='COSINE', use_index=TRUE))
-- ) WHERE source_uri LIKE '%bigquery%';
