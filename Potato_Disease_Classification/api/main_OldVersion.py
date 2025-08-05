from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from sqlalchemy.orm import Session
from datetime import datetime

# Local imports
from .database import get_db, engine, Base
from .models import PredictionResult
from .schemas import PredictionResultResponse
from .config import settings

Base.metadata.create_all(bind=engine)

app = FastAPI()

# When not using TF Serving
#Model = tf.keras.models.load_model("Saved_Models/1")  

# TensorFlow Serving Endpoint : When FastAPI is running locally
#TF_SERVING_URL = "http://localhost:8501/v1/models/potato_disease_classifier:predict"

# When FastAPI is dockerized : 
# TF_SERVING_URL = "http://host.docker.internal:8501/v1/models/potato_disease_classifier:predict"

# When FastAPI and TF Serving are deployed on Render
#TF_SERVING_URL = "http://tf-serving:8501/v1/models/potato_disease_classifier:predict"

#Refactor way :
TF_SERVING_URL = settings.TF_SERVING_URL

# Make sure the model name here matches the MODEL_NAME you used in the Docker command

Class_Names = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file : UploadFile = File(...), db: Session = Depends(get_db)
):
    image = read_file_as_image(await file.read())
    
    image_batch = np.expand_dims(image, 0)
    
    # TensorFlow Serving expects a list of instances or a tensor.
    payload = {"instances": image_batch.tolist()} 
    
    try:
        response = requests.post(TF_SERVING_URL, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        predictions = response.json()["predictions"]
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = Class_Names[predicted_class_index]
        confidence = np.max(predictions[0])
        
        
        db_prediction = PredictionResult(
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=float(confidence) # Ensure float type
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction) # Refresh to get the generated ID and timestamp
        
        return {
            "class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to TensorFlow Serving. Is it running?")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error from TensorFlow Serving: {e}")
    except KeyError:
        raise HTTPException(status_code=500, detail="Invalid response from TensorFlow Serving. 'predictions' key missing.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    
    
    # predictions = Model.predict(image_batch)  #When not using TF Serving
    # predicted_class = Class_Names[np.argmax(predictions[0])]  #When not using TF Serving
    # confidence = np.max(predictions[0])   #When not using TF Serving
    
    # return {
    #     "class": predicted_class,             #When not using TF Serving
    #     "confidence": float(confidence)
    # }
    
@app.get("/history", response_model=list[PredictionResultResponse])
async def get_history(db: Session = Depends(get_db)):
    results = db.query(PredictionResult).order_by(PredictionResult.timestamp.desc()).all()
    return results
    
    
