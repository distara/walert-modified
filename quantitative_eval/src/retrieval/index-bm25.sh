# Stop immediately if a command fails.
set -e

# PURP0SE:
# Build our Walert's sparse BM25/Lucene index.
#
# Input:
#   FAQ passages formatted as JSONL.
#
# Output:
#   target/indexes/bm25/
#
# BM25 is only a RETRIEVER (it ranks passages that may answer a question)
# Does not use Falcon and does not generate the final conversational answer.


# Find this script's own directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# index-bm25.sh lives at:
#   quantitative_eval/src/retrieval/index-bm25.sh
# so two directories above SCRIPT_DIR is quantitative_eval/.
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DIR="${ROOT_DIR}/target/repro/prepared/bm25"
INDEX_DIR="${ROOT_DIR}/target/repro/indexes/bm25"


echo "Building Walert BM25 index"
echo "Input:  ${INPUT_DIR}"
echo "Output: ${INDEX_DIR}"


# Rebuilding should start from a clean generated index.
rm -rf "${INDEX_DIR}"

mkdir -p "$(dirname "${INDEX_DIR}")"


python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${INPUT_DIR}" \
  --language en \
  --index "${INDEX_DIR}" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw