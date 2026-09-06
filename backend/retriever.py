STOP_WORDS = {
    "what", "is", "the", "a", "an",
    "of", "to", "in", "for", "and"
}


def retrieve(query: str):

    with open("document.txt", "r") as f:
        document = f.read()

    query_words = set(query.lower().split())

    # Remove common words
    query_words = query_words - STOP_WORDS

    results = []

    for paragraph in document.split("\n\n"):

        paragraph_words = set(paragraph.lower().split())

        score = len(query_words & paragraph_words)

        if score > 0:
            results.append((score, paragraph))

    results.sort(reverse=True, key=lambda x: x[0])

    return [paragraph for score, paragraph in results[:3]]


if __name__ == "__main__":

    results = retrieve("What is RAG?")

    print("🔎 RETRIEVED:\n")

    for result in results:
        print(result)
        print("-" * 50)