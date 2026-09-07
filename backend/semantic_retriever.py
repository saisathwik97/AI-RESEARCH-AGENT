from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_chunks(documents):

    chunks = []

    for document in documents:

        source = document["source"]
        text = document["text"]

        lines = text.splitlines()

        current_chunk = []

        for line in lines:

            if line.strip() and line[0].isdigit() and ". " in line:

                if current_chunk:

                    chunks.append({
                        "source": source,
                        "text": "\n".join(current_chunk)
                    })

                current_chunk = [line]

            else:

                if line.strip():
                    current_chunk.append(line)

        if current_chunk:

            chunks.append({
                "source": source,
                "text": "\n".join(current_chunk)
            })

    return chunks


def load_documents():
    documents = []

    documents_folder = "documents"

    for filename in os.listdir(documents_folder):

        if filename.endswith(".txt"):

            path = os.path.join(
                documents_folder,
                filename
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:
                documents.append({
                    "source": filename,
                    "text": f.read()
                })

    return documents

def load_chunks():
    document = load_documents()

    chunks = create_chunks(document)

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Chunks saved!")

    return chunks


chunks = load_chunks()


if os.path.exists("embeddings.npy"):

    embeddings = np.load("embeddings.npy")

    print("Embeddings loaded from file!")

else:

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    np.save("embeddings.npy", embeddings)

    print("Embeddings created and saved!")
STOP_WORDS = {
    "what", "is", "the", "a", "an",
    "of", "to", "in", "for", "and"
}
def clean_query(query):
    words = query.lower().split()

    filtered_words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)
def retrieve(query, top_k=3):

    cleaned_query = clean_query(query)

    print(
        "🔎 Original query:",
        query
    )

    print(
        "🧹 Cleaned query:",
        cleaned_query
    )

    query_embedding = model.encode(
        [cleaned_query]
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "similarity": float(
                similarities[index]
            ),

            "source": chunks[index]["source"],

            "text": chunks[index]["text"]
        })

    return results


