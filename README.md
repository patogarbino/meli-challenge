# 🛒 MeLi New/Used Classifier

Clasificador de artículos de MercadoLibre como **nuevos** o **usados** usando CatBoost, con arquitectura hexagonal, API REST y frontend Streamlit.

| Métrica | Valor |
|---------|-------|
| Accuracy | **91.4%** |
| AUC-ROC | **0.9734** |

---

## 📁 Estructura del proyecto

```
challenge-MELI/
├── data/
│   ├── raw/                          # Datos crudos (JSONL original)
│   │   ├── MLA_100k_checked_v3.jsonlines  # Dataset completo (100k items)
│   │   └── MLA_10k_test.jsonlines         # Subset de prueba (10k items)
│   └── processed/                    # Datos procesados (parquets)
│
├── models/                           # Artefactos del modelo entrenado
│   ├── catboost_model.cbm            # Modelo CatBoost serializado
│   └── preprocessor.pkl              # Preprocesador + warranty classifier
│
├── notebooks/                        # Jupyter notebooks de exploración
│   ├── 1.0-pg-EDA.ipynb              # Análisis exploratorio completo
│   └── 3.0-pg-exp-1-catboost.ipynb   # Experimento CatBoost + Optuna
│
├── production/                       # Código de producción (arq. hexagonal)
│   ├── domain/                       # Capa de DOMINIO
│   │   ├── entities.py               #   Entidades: RawItem, TrainResult
│   │   └── ports.py                  #   Interfaces abstractas (ABCs)
│   │
│   ├── adapters/                     # Capa de ADAPTADORES
│   │   ├── data_loader.py            #   Carga JSONL + split train/test
│   │   ├── preprocessor.py           #   Feature engineering completo
│   │   ├── catboost_model.py         #   CatBoost + Optuna
│   │   └── model_repository.py       #   Save/load modelo a disco
│   │
│   ├── application/                  # Capa de APLICACIÓN (use cases)
│   │   ├── train_use_case.py         #   Orquesta el flujo de entrenamiento
│   │   └── predict_use_case.py       #   Orquesta el flujo de predicción
│   │
│   ├── modeling/                     # Entry points CLI
│   │   ├── train.py                  #   CLI para entrenar
│   │   └── predict.py                #   CLI para predecir
│   │
│   ├── api.py                        # FastAPI — endpoint /api/predict
│   ├── warranty_classifier.py        # Clasificador de garantías (NLP)
│   └── new_or_used.py                # Función build_dataset() original
│
├── tests/                            # Tests unitarios e integración
│   ├── test_preprocessor.py          # 12 tests del preprocesador
│   └── test_end_to_end.py            # 4 tests end-to-end
│
├── streamlit_app.py                  # Frontend Streamlit (con login)
├── warranty_samples.csv              # Datos etiquetados para warranty
├── Dockerfile                        # Imagen Docker para Cloud Run
├── nginx.conf                        # Reverse proxy (Streamlit + FastAPI)
├── supervisord.conf                  # Gestor de procesos
├── requirements.txt                  # Dependencias Python
└── .dockerignore                     # Exclusiones del build Docker
```

---

## 🚀 Levantar el proyecto

### Opción 1: Docker (recomendado)

```bash
# Build de la imagen
docker build -t meli-classifier .

# Ejecutar el contenedor
docker run -p 8080:8080 meli-classifier
```

Abrir **http://localhost:8080** → login con `admin` / `Meli`.

### Opción 2: Local (sin Docker)

```bash
# Crear entorno virtual
pyenv virtualenv 3.11.4 meli-challenge
pyenv activate meli-challenge
pip install -r requirements.txt

# Terminal 1 — API
PYTHONPATH=. MODEL_DIR=models/ uvicorn production.api:app --port 8000

# Terminal 2 — Frontend
PYTHONPATH=. API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

- **API**: http://localhost:8000/docs (Swagger)
- **Frontend**: http://localhost:8501 → login con `admin` / `Meli`

---

## 🔌 Uso de la API

### Health check

```bash
curl http://localhost:8000/api/health
```

### Predicción

```bash
curl -F "file=@data/raw/MLA_100k_checked_v3.jsonlines" http://localhost:8000/api/predict
```

Respuesta:

```json
{
  "total_items": 100000,
  "summary": {"new": 53758, "used": 46242},
  "predictions": [
    {"index": 0, "prediction": "new", "prob_new": 0.88, "title": "Auriculares..."},
    ...
  ]
}
```

---

## 🖥️ Uso del Frontend

1. Abrir la app → login con `admin` / `Meli`
2. Subir un archivo `.jsonlines` (se puede usar `data/raw/MLA_100k_checked_v3.jsonlines` como archivo de prueba)
3. Ver el resumen de predicciones con métricas y tabla detallada
4. Filtrar por "nuevos" o "usados", ordenar por probabilidad
5. Descargar resultados como CSV

---

## 🏋️ Entrenamiento

```bash
# Sin optimización de hiperparámetros
PYTHONPATH=. python -m production.modeling.train \
    --data-path data/raw/MLA_100k_checked_v3.jsonlines \
    --warranty-samples warranty_samples.csv \
    --output-dir models/

# Con Optuna (50 trials)
PYTHONPATH=. python -m production.modeling.train \
    --data-path data/raw/MLA_100k_checked_v3.jsonlines \
    --warranty-samples warranty_samples.csv \
    --output-dir models/ \
    --optimize
```

---

## 🧪 Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

16 tests: 12 unitarios (preprocesador) + 4 de integración (pipeline completo).

---

## ☁️ Deploy a Cloud Run

```bash
gcloud run deploy meli-classifier \
    --source . \
    --region us-central1 \
    --port 8080 \
    --memory 2Gi \
    --allow-unauthenticated
```

---

## 🏗️ Arquitectura Hexagonal

```
 ┌──────────────────────────────────────────────┐
 │                  Entry Points                │
 │      (CLI train/predict, FastAPI, Streamlit)  │
 └──────────────────┬───────────────────────────┘
                    │
 ┌──────────────────▼───────────────────────────┐
 │              Application Layer               │
 │        (TrainUseCase, PredictUseCase)         │
 └──────────────────┬───────────────────────────┘
                    │
 ┌──────────────────▼───────────────────────────┐
 │                Domain Layer                  │
 │    (Ports/Interfaces, Entities)              │
 └──────────────────┬───────────────────────────┘
                    │
 ┌──────────────────▼───────────────────────────┐
 │              Adapters Layer                  │
 │  (DataLoader, Preprocessor, CatBoost, Repo)  │
 └──────────────────────────────────────────────┘
```

Los adaptadores implementan las interfaces (ports) definidas en el dominio, permitiendo cambiar cualquier componente sin modificar la lógica de negocio.
