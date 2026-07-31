from embeddings.llama_embeddings import embed_text

vector = embed_text(
    "How do I become a rabbit?"
)

print("Embedding length:", len(vector))
print("First 10 values:")
print(vector[:10])