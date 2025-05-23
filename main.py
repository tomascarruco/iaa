#!/usr/bin/env python
 
# conda activate project_venv
# 
from contextlib import contextmanager
from multiprocessing import Queue
from posix import times
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.quantization import quantize_embeddings
import numpy as np
import torch
import time

MAX_CORPUS_SIZE = 1_000

if __name__ == "__main__":
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    content = []
    with open("./poems/You Are Old, Father William - Lewis Carroll","r") as f:
        content = f.readlines()

    corpus_sentances = set()
    for row in content:
        corpus_sentances.add(row)
        if len(corpus_sentances) > MAX_CORPUS_SIZE:
            break

    corpus_sentances = list(corpus_sentances)
    corpus_embeddings = model.encode(corpus_sentances, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    queries = [
        "Fatherhood",
    ]

    top_k = min(5, len(corpus_sentances))
    q_embedded = model.encode(queries[0], convert_to_tensor=True)

    hits = util.semantic_search(query_embeddings=q_embedded, corpus_embeddings=corpus_embeddings, score_function=util.dot_score)

    simillarity_scores = model.similarity(q_embedded, corpus_embeddings)[0]
    scores, indices = torch.topk(simillarity_scores, k=top_k)

    print("\nQuery:", queries[0])
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(corpus_sentances[idx], f"(Score: {score:.4f})")


    
