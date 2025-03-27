from typing import Optional
from fastapi import FastAPI
import pandas as pd
import joblib
import importlib
import DataPreprocessing

importlib.reload(DataPreprocessing)
from DataPreprocessing import TextPreprocessor, TextVectorizer
from joblib import load
from pydantic import BaseModel
from DataModel import DataModel

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

pipeline = load('./assets/pipeline_model.joblib')

# Definir el esquema de entrada
class NewsInput(BaseModel):
    text: str
import sys
print(sys.modules.keys()) 

@app.post("/predict")
def predict_news(data: NewsInput):
    """
    Recibe un texto plano y devuelve si es una noticia falsa o verdadera con porcentaje de certeza.
    """
    # Aplicar preprocesamiento
    processed_text = pipeline['preprocessor'].preprocessor(data.text)

    # Vectorizar el texto
    new_data = pd.DataFrame({'Texto': [processed_text]})
    new_vectorized = pipeline['vectorizer'].transform(new_data)

    # Obtener predicción y probabilidades
    probabilities = pipeline['model'].predict_proba(new_vectorized)[0]
    prediction = "verdadera" if probabilities[1] > probabilities[0] else "falsa"
    confidence = max(probabilities) * 100  # Convertir a porcentaje

    return {"prediction": prediction, "confidence": f"{confidence:.2f}%"}