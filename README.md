# 🌿 Detección de Enfermedades en Hojas de Café

Servicio web basado en Computación en la Nube que detecta enfermedades en hojas de café
mediante un modelo de visión artificial (TensorFlow/Keras) y genera recomendaciones
técnicas de manejo preventivo usando la **API de Groq**.

## 🔗 Demo

- **App desplegada:** `<pega aquí la URL de Streamlit Community Cloud>`
- **Repositorio:** `<pega aquí el enlace del repo de GitHub>`

## 🩺 Enfermedades detectadas

| Clase | Nombre | Estado de datos |
|---|---|---|
| `healthy` | Hoja sana | ✅ Entrenado |
| `rust` | Roya (*Hemileia vastatrix*) | ✅ Entrenado |
| `phoma` | Phoma | ✅ Entrenado |
| `leaf_miner` | Minador (*Leucoptera coffeella*) | ✅ Entrenado |
| `cercospora` | Cercospora / Mancha de Hierro | ⏳ Pendiente (dataset incompleto — ver nota abajo) |
| `red_spider` | Ácaro rojo | ⏳ Pendiente (dataset incompleto — ver nota abajo) |

> **Nota sobre el dataset:** el `Dataset.zip` original incluye referencias a
> `cercospora_v2.0_fotoEstudio.zip`, `red_spider_v2.zip`, `coffee___rust4.zip`,
> `Miner_Prueba.zip` y `Phoma_Prueba.zip`, pero esos archivos son **punteros de Git LFS**
> (~130 bytes cada uno) sin las imágenes reales — el LFS no se resolvió al comprimir el
> dataset. El modelo actual se entrenó con las 4 clases que sí tienen imágenes reales
> (~1,800 fotos en total). El notebook de entrenamiento (`training/train_model.ipynb`)
> ya está preparado para escalar automáticamente si se agregan las clases faltantes.

## 🏗️ Arquitectura

Ver [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) para el diagrama y detalle completo.

Resumen rápido:

```
Usuario → Streamlit (UI + carga de imagen)
            ├── Modelo Keras (MobileNetV2, local) → clase + % confianza
            └── API de Groq (Llama 3.3 70B) → recomendaciones técnicas en JSON
```

## 📁 Estructura del repositorio

```
coffee-disease-detector/
├── app/
│   ├── app.py                    # Aplicación Streamlit (UI + inferencia)
│   ├── groq_service.py           # Integración con la API de Groq
│   ├── requirements.txt          # Dependencias de la app
│   ├── coffee_disease_model.keras  # (generado por el notebook, no incluido aquí)
│   └── class_indices.json          # (generado por el notebook, no incluido aquí)
├── training/
│   └── train_model.ipynb         # Notebook de entrenamiento para Google Colab
├── docs/
│   └── ARQUITECTURA.md           # Documentación de arquitectura
├── .streamlit/
│   └── config.toml               # Tema visual de Streamlit
├── .env.example                  # Plantilla de variables de entorno
├── .gitignore
└── README.md
```

## 🚀 Paso 1 — Entrenar el modelo (Google Colab)

1. Abre `training/train_model.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Activa GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4)`.
3. Ejecuta todas las celdas (`Entorno de ejecución > Ejecutar todo`). Cuando se te pida,
   sube el archivo `Dataset.zip`.
4. Al final el notebook descarga automáticamente:
   - `coffee_disease_model.keras`
   - `class_indices.json`
   - `metrics.json` (accuracy de test, opcional)
5. Copia `coffee_disease_model.keras` y `class_indices.json` dentro de la carpeta `app/`.

> El modelo final (MobileNetV2 con cabezal propio) pesa aproximadamente 10-15 MB,
> por lo que cabe sin problema en un repositorio normal de GitHub (no requiere Git LFS).

## 🔑 Paso 2 — Configurar la API key de Groq

1. Crea una cuenta y una API key en [console.groq.com](https://console.groq.com/keys).
2. **Para correr localmente:** copia `.env.example` a `.env` y coloca tu key:
   ```
   GROQ_API_KEY=gsk_...
   ```
   Y expórtala antes de correr streamlit, por ejemplo:
   ```bash
   export $(cat .env | xargs)
   ```
3. **Para Streamlit Community Cloud:** en el dashboard de tu app, ve a
   `Settings > Secrets` y pega:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

## 💻 Paso 3 — Correr localmente

```bash
cd app
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate

pip install -r requirements.txt

export GROQ_API_KEY=gsk_tu_api_key   # o usa el .env como se explicó arriba

streamlit run app.py
```

La app abrirá en `http://localhost:8501`.

## ☁️ Paso 4 — Desplegar en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público o compartido con el docente).
   - Asegúrate de que `app/coffee_disease_model.keras` y `app/class_indices.json`
     estén incluidos en el commit (no están en `.gitignore`).
2. Entra a [share.streamlit.io](https://share.streamlit.io/) con tu cuenta de GitHub.
3. Clic en **New app**, selecciona el repositorio y la rama.
4. En **Main file path** coloca: `app/app.py`
5. En **Advanced settings > Secrets**, pega tu `GROQ_API_KEY` como se mostró arriba.
6. Clic en **Deploy**. La primera build tarda unos minutos (instala TensorFlow).

## 🧪 Evidencia de integración con Groq

La función `get_disease_guidance()` en `app/groq_service.py` construye un prompt con
la enfermedad detectada y el % de confianza, y llama al modelo `llama-3.3-70b-versatile`
de Groq pidiendo un JSON estructurado con 5 secciones: diferenciación visual, manejo
agronómico, cuándo consultar a un técnico, monitoreo/seguimiento y registro/trazabilidad.
Esas 5 secciones se renderizan como tarjetas numeradas en la interfaz.

## 🛠️ Tecnologías usadas

- **Frontend/Backend:** Streamlit
- **Modelo de visión:** TensorFlow/Keras — MobileNetV2 (transfer learning + fine-tuning)
- **Generación de texto:** API de Groq (Llama 3.3 70B Versatile)
- **Entrenamiento:** Google Colab (GPU T4)
- **Despliegue:** Streamlit Community Cloud
- **Control de versiones:** Git y GitHub

