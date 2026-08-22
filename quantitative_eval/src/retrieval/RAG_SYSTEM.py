from pathlib import Path
import argparse
import logging

import pandas as pd
from pyserini.search.faiss import FaissSearcher

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re

logging.basicConfig(filename='voice_assistant.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Overall System Pipeline
# [Voice Recording locally] -> {User Query in audio format} -> [ASR using Open AI's Whisper] -> {Query in text format} -> \
# [Dense Retrieval] -> {Top 3 Passages} -> [Textual Response Generation using Falcon] -> {System Response in Text Format} -> [TTS] -> {Final Voice Response in audio format} -> [Play Voice Response using Pygame]


def play_voice_response(text):
    # Language in which you want to convert

    language = 'en'
    # Passing the text and language to the engine
    tts = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in a file named 'output.mp3'
    tts.save("response.mp3")
    sound = AudioSegment.from_mp3("response.mp3")
    sound.export("response.wav", format="wav")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("response.wav")
    pygame.mixer.music.play()
    time.sleep(5)

    os.remove("response.wav")
    os.remove("response.mp3")

def get_answer(text):
    # Find the index of 'Answer:'
    answer = ''
    index = text.find('Answer:')
    if index != -1:
        # Extract the text after 'Answer:'
        answer_text = text[index + len('Answer:'):]
        answer = answer_text.strip()

    else:
        answer = "I apologize, I have no knowledge about that"

    return answer


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto")

    return pipeline, tokenizer


MODEL_NAME = "tiiuae/falcon-7b-instruct"

# Practical local generator for Apple Silicon.
#
# The released research system used Falcon-7B-Instruct above.
# This smaller (baby) 4-bit model lets Walert actually run locally on a M4 Mac
LOCAL_MODEL_NAME = (
    "mlx-community/Llama-3.2-3B-Instruct-4bit"
)

# Load Falcon only when generation is actually needed.
# Importing this file should not immediately allocate a huge language model.
PIPELINE = None
TOKENIZER = None

# Locate quantitative_eval from this file
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

COLLECTION = DATA_DIR / "collection.csv"
TOPICS = DATA_DIR / "topics.csv"
GROUNDTRUTH = DATA_DIR / "groundtruth.csv"

# Dense Retrieval
INDEX = (ROOT/ "target"/ "repro"/ "indexes"/ "tct_colbert-v2-hnp-msmarco-faiss")
QUERY_ENCODER = 'facebook/dpr-question_encoder-multiset-base'
# This is intentionally NOT an Out-of-KB classifier.
#
# All 106 Walert evaluation questions scored at least about 67.10
# with this reproduced dense retriever. A much lower score therefore
# indicates a question that is clearly far outside Walert's domain.
#
# This catches obvious cases such as "What is the capital of France?"
# without rejecting legitimate Walert questions.
OBVIOUSLY_OFF_DOMAIN_SCORE = 66.0


OUTPUT_PATH = '../../target/runs/rag-dense-faiss.txt'
RUN = "dense-faiss"

searcher = FaissSearcher(
    str(INDEX),
    QUERY_ENCODER,
)

def get_context_passages( question, return_score=False):
    """
    Retrieve Walert's top three passages.

    The optional score is useful for detecting questions that are
    obviously far outside Walert's knowledge-base domain.
    """
    num_hits = 10
    hits = searcher.search(question, num_hits)
    top_K = 3

    collection_df = pd.read_csv(COLLECTION)
    context_passages = []

    for d in hits[:top_K]:
        temp_passage = list(
            collection_df[collection_df["passage_id"] == d.docid]["passage"])[0]

        context_passages.append(temp_passage)

    if return_score:
        return (context_passages,hits[0].score)

    return context_passages

def build_prompt(question, context):
    """
    Build the Top-3 prompt used by the released Walert system.

    Keeping prompt creation separate means we can inspect and test the
    RAG pipeline without yet loading the large Falcon model.
    """

    static_prompt = (
    "Answer the following question using only the retrieved documents. "
    "Do not use outside knowledge. "
    "Give a direct, concise answer suitable for a virtual assistant. "
    "Do not mention the retrieved documents, the prompt, or that the answer "
    "was synthesized. "
    "Every response must end with exactly one emoticon from this list: "
    "(>w<, >x<, >.<, TwT, ;-;, T~T, >…<, ^.^, •_•, :p). "
    "If the retrieved documents do not contain enough information to answer "
    "the question, answer NA followed by one emoticon from the list."
)

    prompt = (
        static_prompt
        + "\n Question: " + question
        + "\n Document 1: " + context[0]
        + "\n Document 2: " + context[1]
        + "\n Document 3: " + context[2]
        + "\n Answer: "
    )

    return prompt

def load_local_model():
    """
    Load Walert's practical local language model using MLX.

    MLX is used here because it is designed for Apple Silicon and lets
    us run a quantized model without changing the retrieval system.
    """

    from mlx_lm import load

    model, tokenizer = load(
        LOCAL_MODEL_NAME
    )

    return model, tokenizer


def generate_local_answer(
    question,
    context,
    model,
    tokenizer,
):
    """
    Generate an answer from the same retrieved passages used by Walert.

    This uses the maintained Walert prompt but a smaller (baby) local model.
    The original Falcon path remains available separately!!
    """

    from mlx_lm import generate
    from mlx_lm.sample_utils import (
        make_logits_processors,
        make_sampler,
    )

    prompt = build_prompt(
        question,
        context,
    )

    # The released Walert generator used deterministic generation:
    # do_sample=False and temperature=0.0.
    sampler = make_sampler(
        temp=0.0
    )

    # Preserve Walert's small repetition penalty.
    logits_processors = make_logits_processors(
        repetition_penalty=1.03
    )

    answer = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )

    answer = answer.strip()

    # Small local models do not always follow the requested output format:
    # they may write an explanation before or after NA. If the model explicitly
    # chooses NA, preserve that abstention decision instead of returning the
    # surrounding hallucinated text.
    if re.search(r"\bNA\b", answer, flags=re.IGNORECASE):
        return "NA"

    return answer



def generate_answer(question, context, pipeline, tokenizer):
    prompt_base = build_prompt(
        question,
        context,
    )

    gen_answer = pipeline(
        prompt_base,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # max_length=800,
        max_new_tokens=100,
        # top_k=2,
        # max_new_tokens=400,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.0,
        repetition_penalty=1.03)

    return gen_answer

def inspect_rag(question):
    """
    Show what Walert retrieves and what it would send to Falcon c:
    """

    context = get_context_passages(question)

    prompt = build_prompt(question,context)

    print()
    print("Question:")
    print(question)

    print()
    print("Retrieved passages:")

    for number, passage in enumerate(
        context,
        start=1,
    ):
        print()
        print(f"Document {number}:")
        print(passage)

    print()
    print("Falcon prompt:")
    print()
    print(prompt)


# Create a callback function to stop the recording
def callback(indata, frames, time, status):
    print("...")
    audio_data.append(indata.copy())
    if EVENT.is_set():
        print("Recording finished.")
        raise sd.CallbackStop

# Created only when the voice assistant is actually started.
EVENT = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="Run or inspect the Walert RAG system."
    )

    parser.add_argument(
        "--inspect",
        metavar="QUESTION",
        help=(
            "Retrieve the top three passages and show the Falcon prompt "
            "without loading Falcon or starting the voice assistant."
        ),
    )

    parser.add_argument(
    "--answer",
    metavar="QUESTION",
    help=(
        "Retrieve Walert's top three passages and generate "
        "a local text answer."
    ))

    parser.add_argument(
    "--chat",
    action="store_true",
    help=(
        "Start an interactive text conversation and keep "
        "the local language model loaded between questions."
    ))

    args = parser.parse_args()

    if args.inspect:
        inspect_rag(
            args.inspect
        )
        raise SystemExit(0)

    if args.answer:
        question = args.answer

        print()
        print("Retrieving passages...")

        context, top_score = get_context_passages(question, return_score=True)

        print("Loading local language model...")

        local_model, local_tokenizer = (
            load_local_model()
        )

        print("Generating answer...")

        # Only reject questions that are clearly far outside Walert's domain.
        if top_score < OBVIOUSLY_OFF_DOMAIN_SCORE:
            answer = "NA"
        else:
            answer = generate_local_answer(
                question,
                context,
                local_model,
                local_tokenizer,
            )

        print()
        print("Question:")
        print(question)

        print()
        print("Walert answer:")
        print(answer)

        raise SystemExit(0)

    if args.chat:
        print()
        print("Loading Walert local language model...")

        local_model, local_tokenizer = (
            load_local_model()
        )

        print()
        print("Walert is ready c:\nAsk a question or type 'quit' or 'exit' to stop.")

        while True:
            print()

            question = input("You: ").strip()

            if question.lower() in {
                "quit",
                "exit",
            }:
                print("Walert: Byeeeee!!")
                break

            if not question:
                continue

            context, top_score = get_context_passages(question, return_score=True)

            # Only reject questions that are clearly far outside Walert's domain.
            if top_score < OBVIOUSLY_OFF_DOMAIN_SCORE:
                answer = "NA"
            else:
                answer = generate_local_answer(
                    question,
                    context,
                    local_model,
                    local_tokenizer,
                )

            print()
            print("Walert:")
            print(answer)

        raise SystemExit(0)

    # Everything below this point belongs only to the voice interface.
    # (Text-only RAG should not require audio dependencies.)
    import sounddevice as sd
    from scipy.io.wavfile import write
    import threading
    from pydub import AudioSegment
    import numpy as np
    import whisper
    import os
    from gtts import gTTS
    import pygame
    import time

    EVENT = threading.Event()

    logging.info("Starting the voice assistant...")
    # ************ Query Recording ************

    samplerate = 44100  # Standard for most microphones
    channels = 2  # Stereo

    audio_data = []

    # Start the recording in a new thread
    stream = sd.InputStream(callback=callback, channels=channels, samplerate=samplerate)
    with stream:
        # Wait for the user to press Enter
        input()
        EVENT.set()

    # Concatenate the audio data and save it to a temporary WAV file
    audio = np.concatenate(audio_data)
    temp_filename = 'user_voice_query.wav'
    write(temp_filename, samplerate, audio)

    logging.info("User's voice query successfully recorded and store as user_voice_query.wav")

    # ************ ASR ************
    model = whisper.load_model("base")
    result = model.transcribe(temp_filename, fp16=False)
    question = result["text"]

    logging.info(f"User's voice query trasncribed: {question}")

    logging.info(f"Initiating retrieval . . .")
    # ************ Retrieval ************
    RAG_context_passages = get_context_passages(question)

    logging.info(f"Retrieval Completed")
    # Falcon is intentionally loaded here rather than at import time.
    # This lets us test retrieval and prompt construction independently.
    if PIPELINE is None:
        PIPELINE, TOKENIZER = load_model(MODEL_NAME)
    # ************ Response Generation in Text ************
    logging.info(f"Initiating response generation using Falcon")
    llm_result = generate_answer(question, RAG_context_passages, PIPELINE, TOKENIZER)

    response_text = get_answer(llm_result[0]['generated_text'])

    logging.info(f"Response generation completed (text format): {response_text}")
    # ************ Voice Response ************
    logging.info(f"Initiating voice response (audio format)")
    play_voice_response(response_text)

    logging.info(f"Conversation turn completed.")


