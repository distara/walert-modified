from pyserini.search import FaissSearcher
from pyserini import trectools
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.search.lucene import LuceneSearcher

import pandas as pd

DATA_DIR = "../../data"

COLLECTION = DATA_DIR + "/collection.csv"
TOPICS = DATA_DIR + "/topics.csv"
GROUNDTRUTH = DATA_DIR + "/groundtruth.csv"

#Dense Retrieval
INDEX="../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = 'facebook/dpr-question_encoder-multiset-base'
OUTPUT_PATH = '../../target/runs/rag-dense-faiss.txt'
RUN="dense-faiss" 

#BM25
RUN="bm25" 
INDEX_BM25="../../target/indexes/bm25"
OUTPUT_PATH_BM25 = '../../target/runs/rag-bm25.txt'

# PURP0SE:
# Run retrieval over every question in topics.csv and save the rankings
#
# WARNING:
# This contains both dense and BM25 configuration in the
# same file. RUN is first set to "dense-faiss" and then overwritten with
# "bm25", so the released file effectively selects BM25.
#
# It also constructs the dense FaissSearcher before checking whether BM25
# was selected. We are keeping this file for reference while moving these
# choices into explicit commands such as:
#     ./walert search bm25

topics = pd.read_csv(TOPICS)




searcher = FaissSearcher(
    INDEX,
    QUERY_ENCODER
)
tag = "walert.rag."+RUN
num_hits = 100
output_filename = OUTPUT_PATH

if (RUN=="bm25"):
    searcher = LuceneSearcher(INDEX_BM25)
    output_filename=OUTPUT_PATH_BM25

output_writer = get_output_writer(output_filename, OutputFormat('trec'), 'w',
                                      max_hits=num_hits, tag=tag, topics=topics)
with output_writer:
        #batch_topics = list()
        #batch_topic_ids = list()
        for question_id, question in topics[['question_id','question']].values:
            hits = searcher.search(question, num_hits)
            results = [(question_id, hits)]
            
            for question_id, hits in results:
                output_writer.write(question_id, hits)

            results.clear()
