import os
import joblib
from fastapi import FastAPI 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from util import preprocessing

# Load model and vectorizer
model_path = './models'
try:
    model = joblib.load(os.path.join(model_path, 'model.pkl'))
    vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
except Exception as e:
    print(f'Error Occurred: {e}')

app = FastAPI( debug=True )

# Enable CORS
origins = [
    "http://localhost:3000",
    "https://bbc-news-classification-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home route
@app.get("/")
def home():
    return JSONResponse(
        content={"message": "BBC News Classification API"}, 
        status_code=200
        )

# Pydantic model for JSON input
class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def prediction(request: PredictRequest):
    input_text = request.text
    processed_text = preprocessing(input_text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]

    return JSONResponse(
        status_code=200,
        content={
            "ok":True,
            "message": "Prediction is successful!❤️",
            "text": input_text,
            "processed_text": processed_text,
            "prediction":prediction
        }
    )