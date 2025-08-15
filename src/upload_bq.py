
import argparse, pandas as pd, json
from google.cloud import bigquery

def ensure_table(client, project_id, dataset, table):
    dataset_ref = bigquery.DatasetReference(project_id, dataset)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)

    table_ref = dataset_ref.table(table)
    schema = [
        bigquery.SchemaField("doc_id", "STRING"),
        bigquery.SchemaField("chunk_id", "INT64"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("source_uri", "STRING"),
        bigquery.SchemaField("chunk_text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("embedding_dim", "INT64"),
        bigquery.SchemaField("embedding_model", "STRING"),
    ]
    table_obj = bigquery.Table(table_ref, schema=schema)
    try:
        client.get_table(table_ref)
    except Exception:
        client.create_table(table_obj, exists_ok=True)
    return table_ref

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--project_id", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    project_id = args.project_id or cfg["project_id"]
    dataset = args.dataset or cfg["dataset"]
    table = args.table or cfg["table"]

    df = pd.read_parquet(args.in_path)
    if args.limit and args.limit < len(df):
        df = df.iloc[:args.limit].copy()

    client = bigquery.Client(project=project_id)
    table_ref = ensure_table(client, project_id, dataset, table)

    # Load via DataFrame (fast & typed)
    job = client.load_table_from_dataframe(df[["doc_id","chunk_id","title","source_uri","chunk_text","embedding","embedding_dim","embedding_model"]], table_ref)
    job.result()
    print(f"Uploaded {len(df)} rows to {project_id}.{dataset}.{table}")

if __name__ == "__main__":
    main()
