"""End-to-end RAG pipeline."""

from backend.rag.clinical_context import ClinicalContext
from backend.rag.llm import LLM
from backend.rag.prompt_builder import PromptBuilder
from backend.rag.retriever import Retriever


class RAGPipeline:
    """Retrieve relevant context and generate a grounded answer."""

    def __init__(self):

        self.retriever = Retriever()

        self.prompt_builder = PromptBuilder()

        self.llm = LLM()

    def ask(
        self,
        question: str,
        clinical_context: ClinicalContext | None = None,
    ) -> dict:
        """
        Answer a question using Retrieval-Augmented Generation.

        Returns
        -------
        {
            "question": str,
            "answer": str,
            "contexts": list[dict]
        }
        """

        contexts = self.retriever.retrieve(question)

        prompt = self.prompt_builder.build(
            question=question,
            contexts=contexts,
            clinical_context=clinical_context,
        )

        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }