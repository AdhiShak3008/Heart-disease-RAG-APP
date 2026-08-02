from backend.rag.llm import LLM

llm = LLM()

response = llm.generate(
    "What is a heart murmur?"
)

print(response)