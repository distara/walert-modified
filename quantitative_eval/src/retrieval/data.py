import argparse
from pathlib import Path
import pandas as pd

# Find quantitative_eval from this file's actual location.
#
# This means data.py works whether we run it from:
#   quantitative_eval/
# or:
#   quantitative_eval/src/retrieval/
#
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
TARGET_DIR = ROOT / "target"

COLLECTION = DATA_DIR / "collection.csv"
TOPICS = DATA_DIR / "topics.csv"
GROUNDTRUTH = DATA_DIR / "groundtruth.csv"
INTENT_MAPPING = DATA_DIR / "intent_mapping.csv"
WALERT_INTENT = DATA_DIR / "walert_intent_results.csv"


# Generated reproduction files live separately from the authors'
# supplied reference outputs.
REPRO_DIR = TARGET_DIR / "repro"

PREPARED_DIR = REPRO_DIR / "prepared"

QRELS_OUTPUT = PREPARED_DIR / "qrels.txt"

BM25_COLLECTION_DIR = PREPARED_DIR / "bm25"
COLLECTION_OUTPUT = BM25_COLLECTION_DIR / "collection.jsonl"

INTENT_RUN_OUTPUT = REPRO_DIR / "runs" / "walert-intent.txt"

# PURPOSE:
# Prepare files used by the retrieval experiments.
#
# Note that this script does not perform retrieval and does not run Falcon.
#
# Its main jobs are:
#   1. Convert collection.csv into Pyserini JSONL.
#   2. Convert topic/ground-truth mappings into qrels.txt.
#   3. Convert intent-system results into a TREC retrieval run.
#
# also note that in the cleaned Walert workflow, these responsibilities will be exposed
# through ./walert prepare rather than requiring this file to be run c: 


def create_qrels(topics_filename, groundtruth_filename):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)
    
    data = pd.merge(topics, groundtruth, on='topic_id')
    

    data['subtopic'] = 0
    print(data.shape)

    qrels=data[['question_id', 'subtopic', 'passage_id', 'relevance_judgment']]
    print(qrels.head())
   
    QRELS_OUTPUT.parent.mkdir( parents=True, exist_ok=True, )

    qrels.to_csv(QRELS_OUTPUT,sep="\t",index=False, header=False, )
    print(f"Created qrels: {QRELS_OUTPUT}")

def create_pyserini_collection(collection_filename):
    """
    Convert Walert's FAQ collection into the JSONL format
    expected by Pyserini/Lucene.

    This changes the file format, not the knowledge itself c:
    """

    collection = pd.read_csv(collection_filename)

    collection.columns = ["id","contents"]

    BM25_COLLECTION_DIR.mkdir(parents=True,exist_ok=True)

    collection.to_json(COLLECTION_OUTPUT,orient="records",lines=True)

    print(
        f"Created Pyserini collection: {COLLECTION_OUTPUT}"
    )

def create_topics_msmarco_format(topics_filename):
    topics = pd.read_csv(topics_filename)  
    topics = topics[['question_id', 'question']]
    topics.to_csv(DATA_DIR + "/topics.msmarco-format.txt", sep='\t', index=False, header=False)


def parse_walert_run(topics_filename, groundtruth_filename, walert_filename, intent_mapping_filename, collection_filename):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)
    intents = pd.read_csv(intent_mapping_filename)
    collection = pd.read_csv(collection_filename)

    merged = pd.merge(topics, groundtruth, on='topic_id')
    merged_intents = pd.merge(merged, intents, on='question')
    pd.set_option('display.max_columns', None)

    #print(merged_intents.head())
    walert = pd.read_csv(walert_filename)
    walert = pd.merge(topics, walert, on="question")

    runid = "walert_intent"  
    
    with open(OUTPUT_PATH, 'w') as output_writer:
        
        for question_id, question in topics[['question_id','question']].values:
            row = walert[walert['question_id'] == question_id]
            if (row.values.shape[0] == 0):
                "No results found for question: {}".format(question)
                continue
        
            intent = row['actual'].values[0]
            
            if intent != "AMAZON.FallbackIntent":
                #obtain the passages associated with the intent
                passages = merged_intents[merged_intents['intent'] == intent]
                
                if (intent == "Summary"):
                    # create a dummy passage:
                    result = "P_Summary"
                elif (intent == "BTS"):
                    result = "P_BTS"
                elif (intent == "Degree_Type"):
                    result = "P_Degree_Type"
                elif (intent == "Comparison_Bachelors_Associate"):
                    result = "P_Comparison_Bachelors_Associate"
                elif (passages.shape[0] == 0):
                    print("No passages found for intent: {}".format(intent))   
                    #print(row)
                    print(intent)
                else:
                    #pick one and return it as the ranking for the intent
                    result = passages.passage_id.values[0]
                #generate line for TREC format:
                line = "{} Q0 {} 1 1.0 {}\n".format(question_id, result, runid)
                output_writer.write(line)


def main():
    """
    Choose which Walert data-preparation job to run.
    (e.g. python data.py bm25 -> prepares the files needed for BM25,
          python data.py qrels ->  prepares only qrels.txt,
          python data.py collection -> prepares only collection.jsonl
    """

    parser = argparse.ArgumentParser(
        description="Prepare Walert retrieval data."
    )

    parser.add_argument(
        "task",
        choices=["bm25", "qrels", "collection"],
        help="Which preparation task to run.",
    )

    args = parser.parse_args()

    if args.task == "bm25":
        create_qrels(TOPICS, GROUNDTRUTH,)
        create_pyserini_collection(COLLECTION,)

    elif args.task == "qrels":
        create_qrels(TOPICS, GROUNDTRUTH,)

    elif args.task == "collection":
        create_pyserini_collection(COLLECTION,)


if __name__ == "__main__":
    main()


#   create_topics_msmarco_format(TOPICS)

