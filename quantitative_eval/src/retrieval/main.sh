#!/usr/bin/env bash

# ORIGINAL WALERT PIPELINE
#
# This file came from the released Walert code c:
# It shows the authors' intended order:
#
# data.py -> encode/index -> search.py -> eval.py
# Stop immediately if one stage fails.
#
# This is important because we do not want, for example, evaluation to run
# on an old search result if indexing failed.
set -e


# Find this script's own location rather than user's terminal location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


echo
echo "======================================"
echo "Walert retrieval reproduction"
echo "======================================"
echo


# ---------------------------------------------------------------------------
# 1. Prepare the data
# ---------------------------------------------------------------------------
#
# Creates:
#   target/repro/prepared/bm25/collection.jsonl
#   target/repro/prepared/qrels.txt
#
echo "1/7 Preparing retrieval data..."
python "${SCRIPT_DIR}/data.py" bm25


# ---------------------------------------------------------------------------
# 2. Build the BM25/Lucene index
# ---------------------------------------------------------------------------
#
# Creates:
#   target/repro/indexes/bm25/
#
echo
echo "2/7 Building BM25 index..."
bash "${SCRIPT_DIR}/index-bm25.sh"

# ---------------------------------------------------------------------------
# 3. Encode passages for dense retrieval
# ---------------------------------------------------------------------------
#
# Unlike BM25, dense retrieval first turns every passage into a vector.
#
# Creates:
#   target/repro/embeddings/tct_colbert-v2-hnp-msmarco/
#

echo
echo "3/7 Encoding passages for dense retrieval..."
bash "${SCRIPT_DIR}/encode.sh"

# ---------------------------------------------------------------------------
# 4. Build the dense FAISS index
# ---------------------------------------------------------------------------
#
# Takes the passage vectors from encode.sh and makes them searchable.
#
# Creates:
#   target/repro/indexes/tct_colbert-v2-hnp-msmarco-faiss/
#

echo
echo "4/7 Building dense FAISS index..."
bash "${SCRIPT_DIR}/index.sh"

# ---------------------------------------------------------------------------
# 5. Search all Walert evaluation questions
# ---------------------------------------------------------------------------
#
# Creates:
#   target/repro/runs/rag-bm25.txt
#
echo
echo "5/7 Running BM25 retrieval..."
python "${SCRIPT_DIR}/search.py" bm25

# ---------------------------------------------------------------------------
# 6. Search all Walert evaluation questions with dense retrieval
# ---------------------------------------------------------------------------
#
# search.py encodes each question with the released DPR query encoder
# and searches the dense index we rebuilt above.
#
# Creates:
#   target/repro/runs/rag-dense-faiss.txt
#

echo
echo "6/7 Running dense retrieval..."
python "${SCRIPT_DIR}/search.py" dense


# ---------------------------------------------------------------------------
# 7. Evaluate retrieval quality
# ---------------------------------------------------------------------------

QRELS="${ROOT_DIR}/target/repro/prepared/qrels.txt"
BM25_RUN="${ROOT_DIR}/target/repro/runs/rag-bm25.txt"
DENSE_RUN="${ROOT_DIR}/target/repro/runs/rag-dense-faiss.txt"

echo
echo "7/7 Evaluating BM25 on Known questions..."
python "${SCRIPT_DIR}/eval.py" \
    known \
    "${QRELS}" \
    "${BM25_RUN}"

echo
echo "Evaluating BM25 on Inferred questions..."
python "${SCRIPT_DIR}/eval.py" \
    inferred \
    "${QRELS}" \
    "${BM25_RUN}"

echo
echo "Evaluating dense retrieval on Known questions..."
python "${SCRIPT_DIR}/eval.py" \
    known \
    "${QRELS}" \
    "${DENSE_RUN}"

echo
echo "Evaluating dense retrieval on Inferred questions..."
python "${SCRIPT_DIR}/eval.py" \
    inferred \
    "${QRELS}" \
    "${DENSE_RUN}"

echo
echo "Walert retrieval pipeline complete c:"