"""
Integración con la API de Groq para generar orientación técnica
sobre la enfermedad detectada en la hoja de café.

Requiere la variable de entorno / secret GROQ_API_KEY.
En Streamlit Community Cloud: Settings > Secrets:
    GROQ_API_KEY = "gsk_..."
En local: crea un archivo .env o exporta la variable antes de correr streamlit.
"""

import json
import os

import streamlit as st
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Eres un ingeniero agrónomo experto en fitopatología del café, \
similar a un técnico de campo de IHCAFE (Instituto Hondureño del Café). \
Respondes siempre en español, con lenguaje técnico pero claro para un caficultor. \
Debes responder ÚNICAMENTE con un arreglo JSON válido, sin texto adicional, \
sin backticks de markdown, sin preámbulo."""

USER_PROMPT_TEMPLATE = """Se detectó la siguiente condición en una hoja de café mediante \
un modelo de visión artificial:

- Enfermedad/condición: {label}
- Nombre científico: {scientific}
- Confianza del modelo: {confidence:.1f}%

Genera EXACTAMENTE 5 secciones de orientación técnica para el caficultor, en este orden:
1. "Diferenciación a simple vista" - cómo reconocer esta condición y con qué se confunde.
2. "Manejo agronómico preventivo y correctivo" - buenas prácticas de manejo, fertilización, \
sombra, y tratamiento (fungicida/insecticida/acaricida específico si aplica).
3. "Consulta a un técnico" - cuándo y por qué es importante escalar a un ingeniero agrónomo \
o técnico de campo (ej. IHCAFE si es Honduras).
4. "Monitoreo y seguimiento" - cada cuánto revisar, en qué época, señales de mejora/empeoramiento.
5. "Registro y trazabilidad" - qué datos debe documentar el caficultor sobre la parcela.

Responde con un JSON de la forma:
[
  {{"title": "Diferenciación a simple vista", "body": "..."}},
  {{"title": "Manejo agronómico preventivo y correctivo", "body": "..."}},
  {{"title": "Consulta a un técnico", "body": "..."}},
  {{"title": "Monitoreo y seguimiento", "body": "..."}},
  {{"title": "Registro y trazabilidad", "body": "..."}}
]

Cada "body" debe tener 2-4 oraciones, específicas y accionables. No uses markdown dentro del texto."""


def _get_api_key() -> str | None:
    # Prioridad: Streamlit secrets (deploy) -> variable de entorno (local)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def _fallback_guidance(label: str) -> list[dict]:
    """Se usa solo si la API de Groq falla, para que la app no se rompa en vivo."""
    return [
        {
            "title": "Diferenciación a simple vista",
            "body": f"No fue posible generar la orientación con IA en este momento. "
                    f"Se detectó: {label}. Consulta a un técnico para confirmar el diagnóstico.",
        },
        {
            "title": "Manejo agronómico preventivo y correctivo",
            "body": "Revisa las condiciones de sombra, humedad y nutrición de la parcela mientras "
                    "se restablece la conexión con el servicio de recomendaciones.",
        },
        {
            "title": "Consulta a un técnico",
            "body": "Ante cualquier duda sobre el manejo, contacta a un ingeniero agrónomo o al "
                    "instituto cafetalero de tu región.",
        },
        {
            "title": "Monitoreo y seguimiento",
            "body": "Vuelve a capturar la imagen en unos minutos para reintentar el análisis con IA.",
        },
        {
            "title": "Registro y trazabilidad",
            "body": "Documenta la fecha, ubicación de la parcela y observaciones visuales mientras "
                    "tanto.",
        },
    ]


def get_disease_guidance(label: str, scientific: str, confidence: float) -> list[dict]:
    """Llama a la API de Groq y devuelve una lista de 5 dicts {title, body}."""
    api_key = _get_api_key()
    if not api_key:
        return _fallback_guidance(label)

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        label=label, scientific=scientific, confidence=confidence
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        content = response.choices[0].message.content.strip()

        # Algunos modelos envuelven la respuesta en ```json ... ``` pese a la instrucción.
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)

        # Si el modelo devolvió un objeto envolviendo el arreglo, intenta extraerlo.
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    parsed = value
                    break

        if isinstance(parsed, list) and len(parsed) >= 1:
            return parsed

        return _fallback_guidance(label)

    except Exception as e:
        st.toast(f"⚠️ No se pudo contactar a Groq: {e}", icon="⚠️")
        return _fallback_guidance(label)
