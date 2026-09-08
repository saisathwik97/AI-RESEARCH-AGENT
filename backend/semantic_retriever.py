from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ingest import ingest_documents

import numpy as np
import os
import json
import hashlib
import string


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# LOAD DOCUMENTS AND CREATE CHUNKS
# --------------------------------------------------

chunks = ingest_documents()


# --------------------------------------------------
# CREATE HASH OF DOCUMENT CHUNKS
# --------------------------------------------------

def get_chunks_hash(chunks):

    text = json.dumps(
        chunks,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# STOP WORDS
# --------------------------------------------------

STOP_WORDS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and"
}


# --------------------------------------------------
# CLEAN QUERY
# --------------------------------------------------

def clean_query(query):

    query = query.lower()

    # Replace punctuation with spaces
    # instead of deleting punctuation.
    query = query.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    words = query.split()

    filtered_words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


# --------------------------------------------------
# AUTOMATIC EMBEDDING MANAGEMENT
# --------------------------------------------------

chunks_hash = get_chunks_hash(chunks)

hash_file = "embeddings.hash"


if (
    os.path.exists("embeddings.npy")
    and os.path.exists(hash_file)
):

    # Load previously saved hash
    with open(
        hash_file,
        "r"
    ) as f:

        saved_hash = f.read()


    # Documents have not changed
    if saved_hash == chunks_hash:

        embeddings = np.load(
            "embeddings.npy"
        )

        print(
            "Embeddings loaded from file!"
        )


    # Documents have changed
    else:

        print(
            "Documents changed. "
            "Rebuilding embeddings..."
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = model.encode(
            texts
        )

        np.save(
            "embeddings.npy",
            embeddings
        )

        with open(
            hash_file,
            "w"
        ) as f:

            f.write(chunks_hash)

        print(
            "Embeddings recreated and saved!"
        )


# --------------------------------------------------
# FIRST TIME EMBEDDING CREATION
# --------------------------------------------------

else:

    print(
        "Creating embeddings "
        "for the first time..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts
    )

    np.save(
        "embeddings.npy",
        embeddings
    )

    with open(
        hash_file,
        "w"
    ) as f:

        f.write(chunks_hash)

    print(
        "Embeddings created and saved!"
    )


# --------------------------------------------------
# SEMANTIC RETRIEVAL
# --------------------------------------------------

def retrieve(
    query,
    top_k=3,
    threshold=0.35
):

    cleaned_query = clean_query(query)

    # repr() helps us see hidden spaces/characters
    print(
        "🔎 Original query:",
        repr(query)
    )

    print(
        "🧹 Cleaned query:",
        repr(cleaned_query)
    )

    # Convert query into an embedding
    query_embedding = model.encode(
        [cleaned_query]
    )

    # Calculate similarity with every document chunk
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get top K most similar chunks
    top_indices = similarities.argsort()[
        -top_k:
    ][::-1]

    results = []

    for index in top_indices:

        similarity = float(
            similarities[index]
        )

        # Ignore chunks below threshold
        if similarity < threshold:
            continue

        results.append({
            "similarity": similarity,
            "source": chunks[index]["source"],
            "text": chunks[index]["text"]
        })

    return results


# --------------------------------------------------
# TEST RETRIEVAL DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    results = retrieve(
       "What is Retrieval-Augmented Generation?"
    )

    for result in results:

        print(
            "------------------------------------------------------------"
        )

        print(
            "Source:",
            result["source"]
        )

        print(
            "Similarity:",
            result["similarity"]
        )

        print(
            result["text"]
        )