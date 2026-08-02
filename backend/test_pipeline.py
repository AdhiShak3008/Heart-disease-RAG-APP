from backend.inference.predict_patient import load_patient
from backend.inference.predictor import RosaNetPredictor
from backend.rag.pipeline import RAGPipeline


def main():

    patient_id = "2530"

    question = "What does this result mean? Should I be concerned?"

    predictor = RosaNetPredictor()

    recordings = load_patient(patient_id)

    clinical_context = predictor.predict(recordings)

    pipeline = RAGPipeline()

    result = pipeline.ask(
        question=question,
        clinical_context=clinical_context,
    )

    print("\nClinical AI Analysis")
    print("=" * 60)
    print(f"Patient ID : {patient_id}")
    print(f"Prediction : {clinical_context.prediction}")
    print(f"Confidence : {clinical_context.confidence:.2%}")

    print("\nClass Probabilities")

    for label, probability in clinical_context.probabilities.items():
        print(f"  {label:<8}: {probability:.2%}")

    print("\nQuestion")
    print("=" * 60)
    print(result["question"])

    print("\nAnswer")
    print("=" * 60)
    print(result["answer"])

    print("\nRetrieved Sources")
    print("=" * 60)

    for i, context in enumerate(result["contexts"], start=1):

        print(f"\nContext {i}")
        print(f"Document : {context['title']}")
        print(f"Section  : {context.get('section', 'Unknown')}")
        print(f"Source   : {context['source']}")
        print(f"Score    : {context['score']:.4f}")


if __name__ == "__main__":
    main()