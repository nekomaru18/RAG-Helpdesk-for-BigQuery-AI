# eval_logger.py — Together AI version (query embeddings via SDK)
import argparse, json, pandas as pd, numpy as np, time, time as _t
from sklearn.metrics.pairwise import cosine_similarity
from together import Together
import requests

TOGETHER_API_KEY = "<REDACTED_KEY>"  
MODEL_NAME = "togethercomputer/m2-bert-80M-32k-retrieval"
EMBED_DIM = 768  
MAX_RETRY = 3
RETRY_BACKOFF_S = 2.0

def mrr_at_k(ranked_doc_ids, gold, k=5):
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0

def recall_at_k(ranked_doc_ids, gold, k=5):
    return len(set(ranked_doc_ids[:k]) & set(gold)) / max(1, len(gold))

def embed_query(client: Together, text: str) -> np.ndarray:
    """Embed satu query text via Together + retry sederhana."""
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.embeddings.create(model=MODEL_NAME, input=[text])
            vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
            if vec.shape[0] != EMBED_DIM:
                raise RuntimeError(
                    f"Dimensi embedding {vec.shape[0]} != {EMBED_DIM}. "
                    "Pastikan MODEL_NAME/EMBED_DIM sesuai dengan dokumen."
                )
            return vec
        except Exception as e:
            last_err = e
            # backoff untuk rate limit / network
            _t.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Gagal embed query setelah {MAX_RETRY} percobaan: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="jsonl with qid,question,gold_doc_ids")
    ap.add_argument("--embed", required=True, help="embeddings parquet (dokumen)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    if not TOGETHER_API_KEY.strip():
        raise SystemExit("Error: TOGETHER_API_KEY belum diisi di eval_logger.py")

    # Load embedding dokumen
    df = pd.read_parquet(args.embed)
    X = np.vstack(df["embedding"].values).astype(np.float32)  # (N, D)
    doc_ids = df["doc_id"].astype(str).values

    client = Together(api_key=TOGETHER_API_KEY)

    logs = []
    t0_all = time.time()
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["qid"]
            q = item["question"]
            gold = [str(x) for x in item.get("gold_doc_ids", [])]

            t0 = time.time()
            qv = embed_query(client, q).reshape(1, -1)  # (1, D)
            sims = cosine_similarity(qv, X)[0]
            top_idx = sims.argsort()[::-1][:args.topk]
            ranked_doc_ids = [doc_ids[i] for i in top_idx]
            latency_ms = int((time.time() - t0) * 1000)

            logs.append({
                "qid": qid,
                "mrr@5": mrr_at_k(ranked_doc_ids, gold, 5),
                "recall@5": recall_at_k(ranked_doc_ids, gold, 5),
                "latency_ms": latency_ms,
                "top_doc_ids": ranked_doc_ids,
            })

    total_s = time.time() - t0_all
    out_df = pd.DataFrame(logs)
    out_df.to_csv(args.out, index=False)
    print(f"Saved eval log with {len(out_df)} rows -> {args.out} (Elapsed {total_s:.2f}s)")

if __name__ == "__main__":
    main()
