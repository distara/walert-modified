from pathlib import Path

import pandas as pd
from pyserini.search.lucene import LuceneSearcher


# quantitative_eval/
ROOT = Path(__file__).resolve().parents[2]

TOPICS = ROOT / "data" / "topics.csv"
INDEX = ROOT / "target" / "indexes" / "bm25"

# Keep the authors' original run untouched.
OUTPUT = ROOT / "target" / "runs" / "rag-bm25-reproduced.txt"

NUM_HITS = 100
TAG = "walert.rag.bm25"


print(f"Loading BM25 index from:\n  {INDEX}")
searcher = LuceneSearcher(str(INDEX))
topics = pd.read_csv(TOPICS)

print(f"Loaded {len(topics)} Walert questions.")
print(f"Writing reproduced run to:\n  {OUTPUT}\n")

with OUTPUT.open("w", encoding="utf-8") as out:
    for n, row in topics.iterrows():
        question_id = str(row["question_id"])
        question = str(row["question"])

        hits = searcher.search(question, NUM_HITS)

        for rank, hit in enumerate(hits, start=1):
            out.write(
                f"{question_id} Q0 "
                f"{hit.docid} "
                f"{rank} "
                f"{hit.score:.6f} "
                f"{TAG}\n"
            )

        print(
            f"[{n + 1:3d}/{len(topics)}] "
            f"{question_id}: {len(hits)} hits"
        )

print("\nFinished.")
