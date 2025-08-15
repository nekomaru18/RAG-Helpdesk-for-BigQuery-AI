# src/retriever_local_together.py
import argparse, pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from together import Together

# ==== SAMAKAN DENGAN embed.py ====
TOGETHER_API_KEY = "<REDACTED_KEY>"
MODEL_NAME = "togethercomputer/m2-bert-80M-32k-retrieval"
EMBED_DIM = 768  # samakan dengan kolom embedding_dim di BigQuery
# =================================

def embed_query(client: Together, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL_NAME, input=[text])
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return vec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="parquet embeddings lokal")
    ap.add_argument("--query", required=True, help="teks query")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    X = np.vstack(df["embedding"].values).astype(np.float32)           # (N, D)
    doc = df[["doc_id","chunk_id","title","source_uri","chunk_text"]]  # meta

    client = Together(api_key=TOGETHER_API_KEY)
    qv = embed_query(client, args.query).reshape(1, -1)                # (1, D)

    sims = cosine_similarity(qv, X)[0]
    top_idx = sims.argsort()[::-1][:args.topk]
    out = doc.iloc[top_idx].copy()
    out["score"] = sims[top_idx]
    # tampilkan ringkas
    pd.set_option("display.max_colwidth", 90)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
