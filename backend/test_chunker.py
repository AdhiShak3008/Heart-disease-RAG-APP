from backend.rag.chunker import chunk_text

text = (
    "This is a long medical document discussing heart murmurs, "
    "valve disease, cardiac auscultation, and diagnostic procedures."
)

chunks = chunk_text(
    text,
    chunk_size=40,
    overlap=10,
)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}")
    print(chunk)
