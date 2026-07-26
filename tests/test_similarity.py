from embeddings.llama_embeddings import embed_text
import math


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    return dot / (mag_a * mag_b)


tests = [
    "Item: Bunny Juice (Brown Fur). Transforms you into a rabbit.",
    "How do I become a rabbit?",
    "How do I make butter?"
]

vectors = []

for text in tests:
    print("Embedding:", text)
    vectors.append(embed_text(text))


print()
print("Rabbit question vs Bunny Juice:")
print(cosine_similarity(vectors[0], vectors[1]))

print()
print("Rabbit question vs Butter:")
print(cosine_similarity(vectors[2], vectors[1]))