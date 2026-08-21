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

CORPUS="../../data/collection"
INDEX="../../target/indexes/bm25"

 python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $CORPUS \
  --language en \
  --index $INDEX \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw 