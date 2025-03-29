from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import importlib
import DataPreprocessing

from DataPreprocessing import TextPreprocessor, TextVectorizer
from joblib import load
from pydantic import BaseModel
from DataModel import DataModel

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# Configuración CORS actualizada
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL de tu frontend Next.js
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

pipeline = load('./assets/pipeline_model.joblib')

# Definir el esquema de entrada
class NewsInput(BaseModel):
    Titulo: str      # Campo para el título
    Descripcion: str # Campo para la descripción

@app.post("/predict")
def predict_news(data: NewsInput):
    """
    Recibe un título y descripción, y devuelve si es una noticia falsa o verdadera.
    """
    # Vectorizar el texto
    new_data = pd.DataFrame({
        'Titulo': [data.Titulo],
        'Descripcion': [data.Descripcion]
    })
    
    processed_data = pipeline['preprocessor'].transform(new_data)
    new_vectorized = pipeline['vectorizer'].transform(processed_data)

    # Obtener predicción y probabilidades
    probabilities = pipeline['model'].predict_proba(new_vectorized)[0]
    prediction = "verdadera" if probabilities[1] > probabilities[0] else "falsa"
    confidence = max(probabilities) * 100  # Convertir a porcentaje

    return {"prediction": prediction, "confidence": f"{confidence:.2f}%"}
 

class RetrainInput(BaseModel):
   data: list
 
@app.post("/reentrenar/")
def reentrenar(datos: RetrainInput):
    try:
        # 1. Cargar datos originales
        data_original = pd.read_csv("./data/fake_news_spanish.csv", sep=';', encoding='utf-8')
        
        # 2. Convertir nuevos datos y asegurar que Label sea numérico
        nuevos_datos = pd.DataFrame(datos.data)
        nuevos_datos['Label'] = pd.to_numeric(nuevos_datos['Label'], errors='coerce')  # Conversión a numérico
        
        # 3. Eliminar filas con Label inválido (NaN después de la conversión)
        nuevos_datos = nuevos_datos.dropna(subset=['Label'])
        
        # 4. Combinar ambos conjuntos
        datos_combinados = pd.concat([data_original, nuevos_datos], ignore_index=True)
        
        # 5. Verificar columnas y tipos
        if not {'Titulo', 'Descripcion', 'Label'}.issubset(datos_combinados.columns):
            raise HTTPException(
                status_code=400,
                detail="Las columnas requeridas son: 'Titulo', 'Descripcion', 'Label'"
            )
        
        # 6. Asegurar que Label sea int (0 o 1)
        datos_combinados['Label'] = datos_combinados['Label'].astype(int)
        # 5. Preprocesar (usando el mismo pipeline)
        datos_preprocesados = pipeline['preprocessor'].transform(datos_combinados)
        
        # 6. Dividir datos (stratify para mantener balance de clases)
        X_train, X_test, Y_train, Y_test = train_test_split(
            datos_preprocesados['Texto'],
            datos_preprocesados['Label'],
            test_size=0.3,
            stratify=datos_preprocesados['Label'],
            random_state=1
        )
        
        # 7. Vectorizar (usando el mismo vectorizador, pero ajustado a los nuevos datos)
        X_tfidf_train = pipeline['vectorizer'].fit_transform(X_train)  # ¡Atención: fit_transform!
        X_tfidf_test = pipeline['vectorizer'].transform(X_test)
        
        # 8. Reentrenar modelo
        modelo_actualizado = DecisionTreeClassifier(random_state=1)
        modelo_actualizado.fit(X_tfidf_train, Y_train)
        
        # 9. Evaluar
        Y_pred = modelo_actualizado.predict(X_tfidf_test)
        metricas = {
            "precision": round(precision_score(Y_test, Y_pred, average='weighted'), 4),
            "recall": round(recall_score(Y_test, Y_pred, average='weighted'), 4),
            "f1_score": round(f1_score(Y_test, Y_pred, average='weighted'), 4)
        }
        
        # 10. Actualizar y guardar el pipeline
        pipeline['model'] = modelo_actualizado
        joblib.dump(pipeline, "./assets/pipeline_model.joblib")
        
        return {
            "mensaje": "Modelo reentrenado con datos originales + nuevos",
            "muestras_totales": len(datos_combinados),
            "metricas": metricas
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reentrenar: {str(e)}")