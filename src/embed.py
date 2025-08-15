import argparse, pandas as pd, numpy as np
from tqdm import tqdm
import pyarrow as pa, pyarrow.parquet as pq
from together import Together

TOGETHER_API_KEY = "<REDACTED_KEY>"

MODEL_NAME = "togethercomputer/m2-bert-80M-32k-retrieval"
EMBED_DIM = 768  # dimensi output model ini (cek di dokumentasi Together AI)

def get_embedder():
    if not TOGETHER_API_KEY.strip():
        raise SystemExit("Error: API key Together AI belum diisi di variabel TOGETHER_API_KEY.")

    client = Together(api_key=TOGETHER_API_KEY)

    def _encode(texts):
        response = client.embeddings.create(
            model=MODEL_NAME,
            input=texts
        )
        embs = [item.embedding for item in response.data]
        return np.asarray(embs, dtype=np.float32)

    return _encode, EMBED_DIM, MODEL_NAME

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--limit", type=int, default=0, help="limit rows untuk quick run")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    if args.limit and args.limit < len(df):
        df = df.iloc[:args.limit].copy()

    encode, dim, model_name = get_embedder()
    texts = df["chunk_text"].tolist()
    embs = []
    for i in tqdm(range(0, len(texts), 64)):
        batch = texts[i:i+64]
        vecs = encode(batch)
        embs.append(vecs)

    X = np.vstack(embs)
    df["embedding"] = X.tolist()
    df["embedding_dim"] = dim
    df["embedding_model"] = model_name
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.out_path)
    print(f"Embedded {len(df)} chunks with {model_name} ({dim}d) via Together AI SDK -> {args.out_path}")

if __name__ == "__main__":
    main()
