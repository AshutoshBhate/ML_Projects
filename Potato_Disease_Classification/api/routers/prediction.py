from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from sqlalchemy.orm import Session
import requests
import numpy as np
from PIL import Image
from io import BytesIO

from api import models, schemas, oauth2
from api.database import get_db
from api.config import settings

router = APIRouter(
    prefix="/predictions",
    tags=['Predictions']
)

Class_Names = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@router.post("/", status_code=status.HTTP_200_OK, response_model=schemas.PredictionResultResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    payload = {"instances": image_batch.tolist()}

    try:
        response = requests.post(settings.tf_serving_url, json=payload)
        response.raise_for_status()
        predictions = response.json()["predictions"]

        predicted_class_index = np.argmax(predictions[0])
        predicted_class = Class_Names[predicted_class_index]
        confidence = float(np.max(predictions[0]))

        db_prediction = models.PredictionResult(
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            user_id=current_user.id 
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        return db_prediction

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to the model server.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@router.get("/", response_model=list[schemas.PredictionResultResponse])
async def get_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):

    results = db.query(models.PredictionResult).filter(
        models.PredictionResult.user_id == current_user.id
    ).order_by(models.PredictionResult.timestamp.desc()).all()
    
    return results