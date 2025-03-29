from typing import Optional
from fastapi import FastAPI, HTTPException
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
        # 1. Cargar datos originales (asumiendo que están en './data/fake_news_spanish.csv')
        data_original = pd.read_csv("./data/fake_news_spanish.csv", sep=';', encoding='utf-8')
        
        # 2. Convertir nuevos datos (recibidos por API) a DataFrame
        nuevos_datos = pd.DataFrame(datos.data)
        
        # 3. Combinar ambos conjuntos
        datos_combinados = pd.concat([data_original, nuevos_datos], ignore_index=True)
        
        # 4. Verificar columnas requeridas
        if not {'Titulo', 'Descripcion', 'Label'}.issubset(datos_combinados.columns):
            raise HTTPException(
                status_code=400,
                detail="Las columnas requeridas son: 'Titulo', 'Descripcion', 'Label'"
            )
        
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