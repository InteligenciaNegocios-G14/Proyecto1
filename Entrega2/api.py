from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from preprocessor import TextPreprocessor, TextVectorizer  
from pydantic import BaseModel

# Cargar el pipeline con el contexto correcto
pipeline = joblib.load("pipeline_model.joblib")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API de Clasificación de Noticias Falsas activa 🚀"}

class NewsInput(BaseModel):
    titulo: str
    descripcion: str

@app.post("/predict/")
def predict_news(news: NewsInput):
    new_data = pd.DataFrame({'Titulo': [news.titulo], 'Descripcion': [news.descripcion]})

    new_data = pipeline['preprocessor'].transform(new_data)

    if 'Texto' not in new_data:
        return {"error": "La columna 'Texto' no se generó correctamente."}

    new_vectorized = pipeline['vectorizer'].transform(new_data['Texto'])
    probabilities = pipeline['model'].predict_proba(new_vectorized)
    prediction = pipeline['model'].predict(new_vectorized)[0]
    confidence = round(max(probabilities[0]) * 100, 2)
    resultado = "verdadera" if prediction == 1 else "falsa"

    return {
        "titulo": news.titulo,
        "descripcion": news.descripcion,
        "prediccion": resultado,
        "confianza": f"{confidence}%",
    }

# Ejecutar con: uvicorn api:app --reload
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
        joblib.dump(pipeline, "pipeline_model.joblib")
        
        return {
            "mensaje": "Modelo reentrenado con datos originales + nuevos",
            "muestras_totales": len(datos_combinados),
            "metricas": metricas
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reentrenar: {str(e)}")