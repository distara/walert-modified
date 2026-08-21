"""
Evaluates Walert retrieval runs.

The Walert test collection contains:
- Inferred-answer topics
- Known-answer topics

Evaluation uses NDCG at cutoffs 1, 3, and 5.
"""

import argparse
import sys

import pandas as pd
from ranx import Qrels, Run, compare


KNOWN_TOPICS = {
    *(f"W{i:02d}" for i in range(1, 21)),
    "W39",
}

INFERRED_TOPICS = {
    *(f"W{i:02d}" for i in range(21, 33)),
}


def topic_id_from_question(question_id):
    """
    Convert a Walert question ID into its topic ID.(e.g. W01Q01 -> W01)
    """
    return str(question_id).split("Q", 1)[0]


def load_qrels(path):
    """Load the Walert TREC qrels file."""

    qrels = pd.read_csv(
        path,
        sep="\t",
        names=["q_id", "iteration", "doc_id", "score"],
        header=None,
    )

    # ranx 0.3.x expects Python/Pandas object columns.
    # Modern Pandas may otherwise use StringDtype.
    qrels["q_id"] = qrels["q_id"].astype("object")
    qrels["doc_id"] = qrels["doc_id"].astype("object")

    qrels["topic_id"] = qrels["q_id"].map(topic_id_from_question)

    return qrels


def select_topic_set(qrels, topic_set):
    """Select Known/Inferred Walert topics."""

    if topic_set == "known":
        allowed_topics = KNOWN_TOPICS
    elif topic_set == "inferred":
        allowed_topics = INFERRED_TOPICS
    else:
        raise ValueError(f"Unknown topic set: {topic_set}")

    selected = qrels[
        qrels["topic_id"].isin(allowed_topics)
    ].copy()

    # Filtering can alter Pandas dtype handling, so enforce these again.
    selected["q_id"] = selected["q_id"].astype("object")
    selected["doc_id"] = selected["doc_id"].astype("object")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Walert retrieval runs."
    )

    parser.add_argument(
        "topic_set",
        choices=["known", "inferred"],
        help="Subset of the Walert evaluation collection.",
    )

    parser.add_argument(
        "qrel",
        help="Path to qrels.txt.",
    )

    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more TREC-format run files.",
    )

    args = parser.parse_args()

    qrels_df = load_qrels(args.qrel)
    selected_qrels = select_topic_set(
        qrels_df,
        args.topic_set,
    )

    qrels = Qrels.from_df(
        selected_qrels,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score",
    )

    runs = [
        Run.from_file(path, kind="trec")
        for path in args.runs
    ]

    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=[
            "ndcg@1",
            "ndcg@3",
            "ndcg@5",
        ],
        max_p=0.01,
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4,
    )

    print(f"{args.topic_set.capitalize()} Topics")
    print(report)

    # Keep the original Walert LaTeX-output behavior.
    print(report.to_latex())


if __name__ == "__main__":
    sys.exit(main())
