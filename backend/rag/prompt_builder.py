"""Prompt builder for Retrieval-Augmented Generation."""

from backend.rag.clinical_context import ClinicalContext


class PromptBuilder:
    """Construct prompts from retrieved medical evidence."""

    SYSTEM_PROMPT = """
You are an AI clinical assistant that explains heart sound analysis.

You may receive:

1. AI model predictions produced by a heart sound classifier.
2. Retrieved medical literature.

The AI prediction is NOT a confirmed medical diagnosis.

Rules:
- Use ONLY the retrieved medical evidence to answer.
- Explain how the retrieved evidence relates to the AI prediction.
- If the retrieved documents do not contain the answer, say you do not know.
- Never invent medical facts.
- Never contradict the retrieved evidence.
- Never present the AI prediction as a confirmed diagnosis.
- If the prediction confidence is low, explain the uncertainty.
- Cite document names whenever possible.
""".strip()

    def build(
        self,
        question: str,
        contexts: list[dict],
        clinical_context: ClinicalContext | None = None,
    ) -> str:
        """
        Build a grounded prompt for the LLM.
        """

        prompt = [self.SYSTEM_PROMPT]

        # ------------------------------------------------------
        # Clinical AI Context
        # ------------------------------------------------------

        if clinical_context is not None:

            prompt.append("\n====================")
            prompt.append("Clinical AI Analysis\n")

            prompt.append(
                f"Prediction : {clinical_context.prediction}"
            )

            prompt.append(
                f"Confidence : {clinical_context.confidence:.2%}"
            )

            if clinical_context.probabilities:

                prompt.append("\nClass Probabilities")

                for label, probability in clinical_context.probabilities.items():

                    prompt.append(
                        f"- {label}: {probability:.2%}"
                    )

            if clinical_context.patient_age is not None:

                prompt.append(
                    f"\nPatient Age : {clinical_context.patient_age}"
                )

            if clinical_context.patient_sex:

                prompt.append(
                    f"Patient Sex : {clinical_context.patient_sex}"
                )

            if clinical_context.recording_site:

                prompt.append(
                    f"Recording Site : {clinical_context.recording_site}"
                )

            if clinical_context.notes:

                prompt.append(
                    f"\nClinical Notes:\n{clinical_context.notes}"
                )

        # ------------------------------------------------------
        # Retrieved Context
        # ------------------------------------------------------

        prompt.append("\n====================")
        prompt.append("Retrieved Medical Evidence\n")

        for i, ctx in enumerate(contexts, start=1):

            prompt.append(
                f"""
Context {i}

Document : {ctx.get("title", "Unknown")}
Section  : {ctx.get("section", "Unknown")}
Source   : {ctx.get("source", "Unknown")}

{ctx["text"]}

--------------------------------------------------------------------------------
""".strip()
            )

        # ------------------------------------------------------
        # User Question
        # ------------------------------------------------------

        prompt.append("\n====================")
        prompt.append("Question\n")
        prompt.append(question)

        prompt.append("\n====================")
        prompt.append("Answer")

        return "\n".join(prompt)