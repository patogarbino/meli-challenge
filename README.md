# 🛒 MeLi New/Used Classifier

Clasificador de artículos de MercadoLibre como **nuevos** o **usados** usando CatBoost, con arquitectura hexagonal, API REST, frontend Streamlit y logueo de experimentos con Mlflow.

El desarrollo del challenge comenzó realizando el análisis exploratorio de datos, limieza, seleección e ingeniería de features. Este punto se puede encontrar en el archivo 1.0-pg-EDA.ipynb  dentro de la carperta notebooks. Dentro del mismo se justifican cada una de las decisiones tomadas.

Luego procedí con el modelado, partiendo de un baseline para luego ir a un algoritmo más complejo. Esto fue trabajando en los notebooks 2.0-pg-baseline.ipynb y 3.0-pg-exp-1-catboost.ipynb. Se utilizó Mlflow para el logueo de los entrenamientos.

En el modelado decidí utilizar la metrica de presición de la variable usado como más importante ya que considero que la peor decisión de negocio que puede tomar el algoritmo es decir que un producto usado es nuevo.

Las mejores métricas obtenidas son las siguiente:

| Métrica | Valor |
|---------|-------|
| Accuracy | **91.4%** |
| precision used | **0.9734** |

Luego procedí a generar una simulación de puesta en producción de este algoritmo desarrollando lo siguiente:

- Backend con fastAPI.
- Frontend con Streamlit.
- Arquitectura hexagonal para el código de producción.
- Dockerización de la solución.
- Resguardo del código en github. https://github.com/patogarbino/meli-challenge
- Despliegue en Cloud Run de GCP con CI/CD al repositorio.

La url de cloud run es https://meli-challenge-1026191712427.europe-west1.run.app/ y será eliminada luego de la evaluación al igual que el repositorio.

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
│   ├── catboost_model.cbm            # Modelo CatBoost serializado (~78 MB)
│   └── preprocessor.pkl              # Preprocesador + warranty classifier (~17 KB)
│
├── notebooks/                        # Jupyter notebooks de exploración
│   ├── 1.0-pg-EDA.ipynb              # Análisis exploratorio, selección e ingeniería de features
│   ├── 2.0-pg-baseline.ipynb         # Modelo baseline y primeros experimentos
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

## 📓 Notebooks

Los notebooks documentan el proceso de exploración y modelado:

| Notebook | Descripción |
|----------|-------------|
| **1.0-pg-EDA.ipynb** | Análisis exploratorio de datos: distribuciones, valores faltantes, correlaciones y visualizaciones |
| **2.0-pg-baseline.ipynb** | Modelo baseline y primeros experimentos de clasificación |
| **3.0-pg-exp-1-catboost.ipynb** | Modelo CatBoost con optimización de hiperparámetros vía Optuna |

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

---

## 🖥️ Uso del Frontend

1. Abrir la app → login con `admin` / `Meli`
2. Subir un archivo `.jsonlines` (usar ``data/raw/MLA_10k_test.jsonlines` como archivos de prueba)
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

## 📊 MLflow — Tracking de Experimentos

Se utilizó [MLflow](https://mlflow.org/) para registrar y comparar los entrenamientos realizados en los notebooks 2.0 y 3.0. Cada run loguea métricas, hiperparámetros y artefactos del modelo.

### Levantar la UI de MLflow

```bash
mlflow ui --backend-store-uri models/mlruns --port 5001
```

Abrir **http://localhost:5001** en el navegador.

### Experimentos registrados

La página principal muestra los experimentos creados: **CatBoost** y **Baseline Model**.

![MLflow Home — Experimentos recientes](docs/images/mlflow_home.png)

### Training Runs

Dentro del experimento **CatBoost** se pueden ver los diferentes runs, incluyendo `catboost_optuna`, `catboost_selected_feat...` y `catboost_baseline`, con su duración y fuente (notebook).

![MLflow — Lista de training runs del experimento CatBoost](docs/images/mlflow_experiment_runs.png)

### Detalle de un Run

Al entrar a un run específico (`catboost_optuna`) se visualizan las **8 métricas** registradas:

| Métrica | Valor |
|---------|-------|
| test_accuracy | **0.91** |
| test_precision_new | 0.931 |
| test_recall_new | 0.899 |
| test_precision_used | 0.887 |
| test_recall_used | 0.922 |
| test_f1_used | 0.904 |
| val_precision_used | 0.901 |
| test_auc | 0.029 |

![MLflow — Detalle de métricas del run catboost_optuna](docs/images/mlflow_run_details.png)

---

## 🧪 Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

16 tests: 12 unitarios (preprocesador) + 4 de integración (pipeline completo).

---

## ☁️ Deploy a Cloud Run

El proyecto se despliega automáticamente a Cloud Run via CI/CD con Cloud Build. Cada push a `main` gatilla los cambios:

1. Cloud Build detecta el cambio y realiza el build de la imagen Docker
2. Realiza pushea la imagen a Artifact Registry
3. Genera el despliegue de una nueva revisión en Cloud Run

---
