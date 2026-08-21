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
echo "Walert BM25 retrieval reproduction"
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
echo "1/4 Preparing retrieval data..."
python "${SCRIPT_DIR}/data.py" bm25


# ---------------------------------------------------------------------------
# 2. Build the BM25/Lucene index
# ---------------------------------------------------------------------------
#
# Creates:
#   target/repro/indexes/bm25/
#
echo
echo "2/4 Building BM25 index..."
bash "${SCRIPT_DIR}/index-bm25.sh"


# ---------------------------------------------------------------------------
# 3. Search all Walert evaluation questions
# ---------------------------------------------------------------------------
#
# Creates:
#   target/repro/runs/rag-bm25.txt
#
echo
echo "3/4 Running BM25 retrieval..."
python "${SCRIPT_DIR}/search.py" bm25


# ---------------------------------------------------------------------------
# 4. Evaluate retrieval quality
# ---------------------------------------------------------------------------

QRELS="${ROOT_DIR}/target/repro/prepared/qrels.txt"
BM25_RUN="${ROOT_DIR}/target/repro/runs/rag-bm25.txt"

echo
echo "4/4 Evaluating Known questions..."
python "${SCRIPT_DIR}/eval.py" \
    known \
    "${QRELS}" \
    "${BM25_RUN}"

echo
echo "Evaluating Inferred questions..."
python "${SCRIPT_DIR}/eval.py" \
    inferred \
    "${QRELS}" \
    "${BM25_RUN}"


echo
echo "BM25 pipeline complete."