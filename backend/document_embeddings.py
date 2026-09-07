from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read document
with open("document.txt", "r") as f:
    document = f.read()

# Split document into paragraphs
paragraphs = document.split("\n")

# Create embeddings
embeddings = model.encode(paragraphs)

print("Number of paragraphs:", len(paragraphs))
print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))
