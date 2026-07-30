from backend.rag.retriever import Retriever

retriever = Retriever()

results = retriever.retrieve("What are the symptoms of heart murmurs?")

for i, result in enumerate(results, start=1):

    print("=" * 70)

    print(f"Rank {i}")

    print(f"Score : {result['score']:.4f}")

    print(f"Source: {result['source']}")

    print()

    print(result["text"][:500])
