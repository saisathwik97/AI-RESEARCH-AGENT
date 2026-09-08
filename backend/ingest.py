import os
import json


DOCUMENTS_FOLDER = "documents"
CHUNKS_FILE = "chunks.json"


def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):

        if filename.endswith(".txt"):

            path = os.path.join(
                DOCUMENTS_FOLDER,
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


def create_chunks(documents):

    chunks = []

    for document in documents:

        source = document["source"]
        text = document["text"]

        lines = text.splitlines()

        current_chunk = []

        for line in lines:

            if (
                line.strip()
                and line[0].isdigit()
                and ". " in line
            ):

                if current_chunk:

                    chunks.append({
                        "source": source,
                        "text": "\n".join(
                            current_chunk
                        )
                    })

                current_chunk = [line]

            else:

                if line.strip():
                    current_chunk.append(line)

        if current_chunk:

            chunks.append({
                "source": source,
                "text": "\n".join(
                    current_chunk
                )
            })

    return chunks


def ingest_documents():

    documents = load_documents()

    chunks = create_chunks(documents)

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Created {len(chunks)} chunks "
        f"from {len(documents)} documents."
    )
    return chunks


if __name__ == "__main__":

    ingest_documents()