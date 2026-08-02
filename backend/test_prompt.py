from backend.rag.prompt_builder import PromptBuilder
from backend.rag.retriever import Retriever

retriever = Retriever()

builder = PromptBuilder()

contexts = retriever.retrieve(
    "What are the symptoms of heart murmurs?"
)

prompt = builder.build(
    question="What are the symptoms of heart murmurs?",
    contexts=contexts,
)

print(prompt)