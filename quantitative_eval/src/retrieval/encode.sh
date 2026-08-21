#!/usr/bin/env bash

set -e

# This script converts Walert's passages into 768-number dense vectors!!
#
# collection.jsonl
#       ↓
# TCT-ColBERT passage encoder
#       ↓
# target/repro/embeddings/
#
# These are PASSAGE vectors (Questions are encoded later during search )

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

#encode
CORPUS="${ROOT_DIR}/data/collection.jsonl"
ENCODER="tct_colbert-v2-hnp-msmarco"
EMBEDDINGS="${ROOT_DIR}/target/repro/embeddings/${ENCODER}"

# ensures embedding consistency (removes old file)
rm -rf "${EMBEDDINGS}"
mkdir -p "$(dirname "${EMBEDDINGS}")"

echo "Encoding Walert passages"
echo "Corpus:     ${CORPUS}"
echo "Embeddings: ${EMBEDDINGS}"

python -m pyserini.encode \
 input   --corpus $CORPUS  \
          --fields text \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings $EMBEDDINGS \
            --to-faiss \
  encoder --encoder castorini/$ENCODER \
          --fields text \
          --batch 32 \
          --device cpu