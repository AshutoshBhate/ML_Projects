from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api import models
from api.routers import prediction, user, auth
from api.database import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Creating database tables...")
    models.Base.metadata.create_all(bind=engine)
    yield
    print("Application shutdown.")

app = FastAPI(
    title="Potato Disease Classifier API",
    description="An API to classify potato leaf diseases and manage user prediction history.",
    lifespan=lifespan 
)

origins = [
    "http://localhost:8502", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(prediction.router)

@app.get("/", tags=['Root'])
def root():
    return {"message": "Welcome to the Potato Disease Classification API. Please visit /docs for the API documentation."}