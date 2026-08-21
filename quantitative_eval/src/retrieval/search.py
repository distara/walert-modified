import argparse
from pathlib import Path

from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.search.lucene import LuceneSearcher

import pandas as pd

# Locate quantitative_eval from this file instead of relying on the
# directory from which the user happens to run Python.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
TARGET_DIR = ROOT / "target"

TOPICS = DATA_DIR / "topics.csv"

# ---------------------------------------------------------------------------
# Retrieval locations
# ---------------------------------------------------------------------------

# Search the index that Walert rebuilt itself rather than relying on
# the pre-generated index shipped with the repository.
BM25_INDEX = (TARGET_DIR/ "repro"/ "indexes"/ "bm25")
BM25_OUTPUT = TARGET_DIR / "repro" /  "runs" / "rag-bm25.txt"

# Dense searches should also use the index rebuilt by encode.sh + index.sh.
DENSE_INDEX = (TARGET_DIR/ "repro"/ "indexes"/ "tct_colbert-v2-hnp-msmarco-faiss")

DENSE_OUTPUT = (TARGET_DIR/ "repro"/ "runs"/ "rag-dense-faiss.txt")

# NOTE:
# The released Walert code uses this DPR query encoder against a dense
# index produced with TCT-ColBERT passage embeddings.
#
# That apparent model mismatch is being preserved for now because we
# do not know  whether it is an error in the release or part of the
# experiment that produced the supplied dense run.
DENSE_QUERY_ENCODER = "facebook/dpr-question_encoder-multiset-base"


def main():
    """
    Run one of Walert's retrieval systems over every question in topics.csv!!

    Examples: python search.py bm25

    Later: python search.py dense

    BM25 and dense retrieval have the same job:
    rank FAQ passages for each question.

    They differ only in HOW they decide which passages are relevant.
    """

    parser = argparse.ArgumentParser(
        description="Run Walert retrieval."
    )

    parser.add_argument(
        "retriever",
        choices=["bm25", "dense"],
        help="Retrieval system to run.",
    )

    parser.add_argument(
        "--hits",
        type=int,
        default=100,
        help="Maximum number of passages to retrieve per question.",
    )

    args = parser.parse_args()

    topics = pd.read_csv(TOPICS)

    # ---------------------------------------------------------------
    # Select the requested retrieval system.
    # ---------------------------------------------------------------

    if args.retriever == "bm25":

        searcher = LuceneSearcher(
            str(BM25_INDEX)
        )

        output_filename = BM25_OUTPUT
        tag = "walert.rag.bm25"

    else:
        # Dense retrieval uses FAISS, but BM25 does not.
        #
        # Import FaissSearcher only when dense retrieval is actually chosen.
        # This prevents a dense-retrieval dependency problem from breaking
        # the completely separate BM25 system.
        from pyserini.search.faiss import FaissSearcher

        searcher = FaissSearcher(
            str(DENSE_INDEX),
            DENSE_QUERY_ENCODER,
        )

        output_filename = DENSE_OUTPUT
        tag = "walert.rag.dense.faiss"

    print(
        f"Retriever: {args.retriever}"
    )

    print(
        f"Questions: {len(topics)}"
    )

    print(
        f"Output: {output_filename}"
    )

    output_filename.parent.mkdir(
    parents=True,
    exist_ok=True,)

    output_writer = get_output_writer(
        str(output_filename),
        OutputFormat("trec"),
        "w",
        max_hits=args.hits,
        tag=tag,
        topics=topics,
    )

    with output_writer:

        for question_id, question in topics[
            ["question_id", "question"]
        ].values:

            hits = searcher.search(
                question,
                args.hits,
            )

            output_writer.write(
                question_id,
                hits,
            )


if __name__ == "__main__":
    main()
