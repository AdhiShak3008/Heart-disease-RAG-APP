from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.inference.audio_preprocessor import prepare_recordings
from backend.inference.predictor import RosaNetPredictor

router = APIRouter()

predictor = RosaNetPredictor()


@router.get("/")
def root():

    return {"message": "Heart Disease AI API"}


@router.get("/health")
def health():

    return {"status": "ok", "model": "RosaNet"}


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

                shutil.copyfileobj(
                    upload.file,
                    buffer,
                )

            file_paths[valve] = destination

        recordings = prepare_recordings(file_paths)

        result = predictor.predict(recordings)

        return result
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
