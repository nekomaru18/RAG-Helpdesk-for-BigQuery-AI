
.PHONY: all extract chunk embed upload query eval

all: extract chunk embed upload

extract:
	python src/extract.py --zip ./dataset_files.zip --out ./artifacts/raw

chunk:
	python src/clean_chunk.py --indir ./artifacts/raw --out ./artifacts/chunks.parquet

embed:
	python src/embed.py --in ./artifacts/chunks.parquet --out ./artifacts/embeddings.parquet

upload:
	python src/upload_bq.py --in ./artifacts/embeddings.parquet --limit 5000

query:
	bq query --use_legacy_sql=false < src/query_bq.sql

eval:
	python src/eval_logger.py --pairs ./artifacts/eval_questions.jsonl --embed ./artifacts/embeddings.parquet --out ./artifacts/eval_log.csv
