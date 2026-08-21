from ranx import compare, Qrels, Run
import argparse
import sys
import pandas as pd


KNOWN_TOPICS = {
    *(f"W{i:02d}" for i in range(1, 21)),
    "W39",
}

INFERRED_TOPICS = {
    *(f"W{i:02d}" for i in range(21, 33)),
}


def get_topic_id(question_id):
    """W01Q04 -> W01, W40Q1 -> W40"""
    return str(question_id).split("Q", 1)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Walert retrieval runs."
    )

    parser.add_argument(
        "topic_set",
        choices=["known", "inferred"],
        help="Subset of the Walert test collection to evaluate.",
    )

    parser.add_argument(
        "qrel",
        help="TREC qrels file.",
    )

    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more TREC run files.",
    )

    args = parser.parse_args()

    alpha = 0.01

    qrels_df = pd.read_csv(
        args.qrel,
        sep="\t",
        names=["q_id", "iteration", "doc_id", "score"],
        header=None,
    )

    # ranx 0.3.x expects Pandas object/string columns.
    # Newer Pandas versions may instead use StringDtype.
    qrels_df["q_id"] = qrels_df["q_id"].astype("object")
    qrels_df["doc_id"] = qrels_df["doc_id"].astype("object")

    qrels_df["topic_id"] = qrels_df["q_id"].map(get_topic_id)

    if args.topic_set == "known":
        selected = qrels_df[
            qrels_df["topic_id"].isin(KNOWN_TOPICS)
        ].copy()
    else:
        selected = qrels_df[
            qrels_df["topic_id"].isin(INFERRED_TOPICS)
        ].copy()

    # Keep ranx-compatible dtypes after filtering as well.
    selected["q_id"] = selected["q_id"].astype("object")
    selected["doc_id"] = selected["doc_id"].astype("object")

    qrels = Qrels.from_df(
        selected,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score",
    )

    runs = [
        Run.from_file(run_path, kind="trec")
        for run_path in args.runs
    ]

    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=[
            "ndcg@1",
            "ndcg@3",
            "ndcg@5",
        ],
        max_p=alpha,
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4,
    )

    print(f"{args.topic_set} Topics")
    print(report)

    try:
        print(report.to_latex())
    except Exception as exc:
        print(f"\nLaTeX output unavailable: {exc}")


if __name__ == "__main__":
    sys.exit(main())