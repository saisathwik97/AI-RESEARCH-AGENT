from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Retrieval-Augmented Generation allows an AI system to retrieve information from documents."

embedding = model.encode(text)

print("Embedding created!")
print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])