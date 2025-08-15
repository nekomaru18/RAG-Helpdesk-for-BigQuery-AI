
import argparse, os, pathlib, re, pandas as pd
from src.utils import normalize_ws, stable_id, guess_title_from_path
from tqdm import tqdm
import pyarrow as pa, pyarrow.parquet as pq

def split_into_chunks(text, max_chars=900, overlap=150):
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=3000)
    args = ap.parse_args()

    rows = []
    for root, _, files in os.walk(args.indir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                fp = os.path.join(root, fn)
                rel = pathlib.Path(os.path.relpath(fp, args.indir)).as_posix()
                raw = open(fp, encoding='utf-8', errors='ignore').read()
                title = guess_title_from_path(rel)
                chunks = split_into_chunks(raw[:args.max_chars*50], args.chunk_size, args.chunk_overlap)
                chunks = [c for c in chunks if len(c) >= args.min_chars]
                doc_id = stable_id(rel)
                for i, c in enumerate(chunks):
                    rows.append({
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "title": title,
                        "source_uri": rel,
                        "chunk_text": c
                    })
    df = pd.DataFrame(rows)
    if not len(df):
        raise SystemExit("No chunks produced. Check input.")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.out)
    print(f"Wrote {len(df)} chunks -> {args.out}")

if __name__ == "__main__":
    main()
