from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from backend.inference.audio_preprocessor import prepare_recordings
from backend.inference.predictor import RosaNetPredictor
from backend.rag.pipeline import RAGPipeline

router = APIRouter()

predictor = RosaNetPredictor()
rag = RAGPipeline()

# Stores the latest analysis for follow-up chat
last_clinical_context = None


@router.get("/")
def root():

    return {"message": "Heart Disease AI API"}


@router.get("/health")
def health():

    return {
        "status": "ok",
        "model": "RosaNet",
    }


@router.post("/predict")
async def predict(
    av: UploadFile = File(...),
    mv: UploadFile = File(...),
    pv: UploadFile = File(...),
    tv: UploadFile = File(...),
):

    temp_dir = Path(tempfile.mkdtemp())

    try:

        uploads = {
            "AV": av,
            "MV": mv,
            "PV": pv,
            "TV": tv,
        }

        file_paths = {}

        for valve, upload in uploads.items():

            destination = temp_dir / upload.filename

            with open(destination, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)

            file_paths[valve] = destination

        recordings = prepare_recordings(file_paths)

        clinical_context = predictor.predict(recordings)

        return clinical_context

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


@router.post("/analyze")
async def analyze(
    av: UploadFile = File(...),
    mv: UploadFile = File(...),
    pv: UploadFile = File(...),
    tv: UploadFile = File(...),
    question: str | None = Form(None),
):

    global last_clinical_context

    temp_dir = Path(tempfile.mkdtemp())

    try:

        uploads = {
            "AV": av,
            "MV": mv,
            "PV": pv,
            "TV": tv,
        }

        file_paths = {}

        for valve, upload in uploads.items():

            destination = temp_dir / upload.filename

            with open(destination, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)

            file_paths[valve] = destination

        recordings = prepare_recordings(file_paths)

        clinical_context = predictor.predict(recordings)

        # Save for future chat
        last_clinical_context = clinical_context

        if question is None:
            question = (
                "Explain the model prediction and summarize the "
                "relevant medical evidence."
            )

        rag_result = rag.ask(
            question=question,
            clinical_context=clinical_context,
        )

        return {
            "prediction": clinical_context.prediction,
            "confidence": clinical_context.confidence,
            "probabilities": clinical_context.probabilities,
            "answer": rag_result["answer"],
            "sources": rag_result["contexts"],
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


@router.post("/chat")
async def chat(
    question: str = Body(..., embed=True),
):

    global last_clinical_context

    if last_clinical_context is None:

        raise HTTPException(
            status_code=400,
            detail="Please analyze heart sound recordings first.",
        )

    rag_result = rag.ask(
        question=question,
        clinical_context=last_clinical_context,
    )

    return {
        "answer": rag_result["answer"],
        "sources": rag_result["contexts"],
    }