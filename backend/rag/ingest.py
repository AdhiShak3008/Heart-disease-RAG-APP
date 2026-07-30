"""Document ingestion pipeline."""

from backend.rag.chunker import chunk_text
from backend.rag.config import DOCUMENT_DIR
from backend.rag.embeddings import EmbeddingModel
from backend.rag.pdf_loader import load_pdf
from backend.rag.vector_store import VectorStore


def ingest_documents() -> None:
    """Ingest all PDFs into the Qdrant vector database."""

    print("=" * 60)
    print("Starting document ingestion...")
    print("=" * 60)

    embedder = EmbeddingModel()

    # Determine embedding dimension automatically
    sample_vector = embedder.embed_query("heart murmur")

    store = VectorStore(
        vector_size=len(sample_vector),
    )

    pdf_files = sorted(DOCUMENT_DIR.rglob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDFs\n")

    ids = []
    embeddings = []
    payloads = []

    current_id = 0

    for pdf in pdf_files:

        print(f"Ingesting: {pdf.relative_to(DOCUMENT_DIR)}")

        text = load_pdf(pdf)

        if not text.strip():
            print("   No text extracted. Skipping.\n")
            continue

        chunks = chunk_text(text)

        if not chunks:
            print("   No chunks generated. Skipping.\n")
            continue

        vectors = embedder.embed_documents(chunks)

        for chunk, vector in zip(chunks, vectors):

            ids.append(current_id)

            embeddings.append(vector)

            payloads.append(
                {
                    "text": chunk,
                    "title": pdf.stem,
                    "source": pdf.parent.name,
                    "path": str(pdf),
                }
            )

            current_id += 1

        print(f"   {len(chunks)} chunks")

    print("\nUploading vectors to Qdrant...")

    store.add_documents(
        ids=ids,
        embeddings=embeddings,
        payloads=payloads,
    )

    print(f"\nSuccessfully stored {len(ids)} chunks.")

    # Close Qdrant client (avoids Windows shutdown warning)
    store.client.close()


if __name__ == "__main__":
    ingest_documents()
