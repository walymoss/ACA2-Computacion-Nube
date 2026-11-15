
import os
import streamlit as st
import pandas as pd
import joblib

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Asistente de Permanencia Educativa",
    page_icon="🎓",
    layout="wide"
)

COLLECTION_NAME = "permanencia_educativa"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Credenciales: variables de entorno o Streamlit Secrets
def get_secret(name):
    return os.getenv(name) or st.secrets.get(name, None)

QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


# =========================================================
# 2. CONEXIONES Y MODELOS
# =========================================================

@st.cache_resource
def conectar_qdrant(url, api_key):
    return QdrantClient(url=url, api_key=api_key)


@st.cache_resource
def cargar_embeddings():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def conectar_gemini(api_key):
    return genai.Client(api_key=api_key)


@st.cache_resource
def cargar_modelo_ml():
    ruta = "modelo_mlp.pkl"

    if os.path.exists(ruta):
        return joblib.load(ruta)

    return None


# Validar configuración
if not all([QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY]):
    st.error(
        "Faltan credenciales. Configura QDRANT_URL, "
        "QDRANT_API_KEY y GEMINI_API_KEY en los Secrets."
    )
    st.stop()

try:
    qdrant = conectar_qdrant(QDRANT_URL, QDRANT_API_KEY)
    embedding_model = cargar_embeddings()
    gemini = conectar_gemini(GEMINI_API_KEY)

except Exception as e:
    st.error(f"Error al inicializar los servicios: {e}")
    st.stop()


# =========================================================
# 3. RECUPERACIÓN SEMÁNTICA DESDE QDRANT
# =========================================================

def recuperar_contexto(pregunta, top_k=3):

    vector = embedding_model.encode(
        pregunta,
        normalize_embeddings=True
    ).tolist()

    resultados = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True
    ).points

    return resultados


# =========================================================
# 4. GENERACIÓN DE RESPUESTAS RAG CON GEMINI
# =========================================================

def generar_respuesta(pregunta):

    resultados = recuperar_contexto(pregunta, top_k=3)

    if not resultados:
        return "No se encontraron documentos relevantes.", []

    contexto = "\n\n".join([
        f"[Fuente: {r.payload.get('archivo', 'Documento sin nombre')}]\n"
        f"{r.payload.get('texto', '')[:1200]}"
        for r in resultados
    ])

    prompt = f"""
    Eres un asistente especializado en permanencia estudiantil.

    Instrucciones:
    - Responde exclusivamente con base en el contexto.
    - No inventes información.
    - Cita los documentos utilizados.
    - Si no hay información suficiente, indícalo claramente.
    - Responde en español y de forma clara.

    CONTEXTO:
    {contexto}

    PREGUNTA:
    {pregunta}
    """

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=500
        )
    )

    fuentes = [
        {
            "archivo": r.payload.get("archivo", "Sin nombre"),
            "texto": r.payload.get("texto", ""),
            "score": r.score
        }
        for r in resultados
    ]

    return response.text, fuentes


# =========================================================
# 5. EXPLORACIÓN DE LA BASE DOCUMENTAL
# =========================================================

@st.cache_data(ttl=300)
def explorar_documentos():

    puntos, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    documentos = []

    for punto in puntos:
        payload = punto.payload or {}

        documentos.append({
            "archivo": payload.get("archivo", "Sin nombre"),
            "texto": payload.get("texto", "")
        })

    return documentos


# =========================================================
# 6. INTERFAZ PRINCIPAL
# =========================================================

st.title("🎓 Asistente de Permanencia Educativa")

st.write(
    "Plataforma de consulta documental y análisis del "
    "rendimiento académico mediante Inteligencia Artificial."
)

tab_chat, tab_documentos, tab_ml = st.tabs([
    "💬 Asistente RAG",
    "📚 Base documental",
    "📊 Predicción académica"
])


# =========================================================
# 7. MÓDULO DE CHAT
# =========================================================

with tab_chat:

    st.subheader("Consulta académica")

    st.caption(
        "Realiza preguntas sobre permanencia, acompañamiento "
        "estudiantil y estrategias institucionales."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for mensaje in st.session_state.messages:

        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

            if mensaje.get("fuentes"):
                with st.expander("Consultar fuentes utilizadas"):
                    for fuente in mensaje["fuentes"]:
                        st.markdown(f"**{fuente['archivo']}**")
                        st.caption(
                            f"Similitud: {fuente['score']:.4f}"
                        )
                        st.write(fuente["texto"][:700])

    pregunta = st.chat_input(
        "Escribe tu pregunta..."
    )

    if pregunta:

        st.session_state.messages.append({
            "role": "user",
            "content": pregunta
        })

        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):

            with st.spinner("Consultando documentos y generando respuesta..."):

                try:
                    respuesta, fuentes = generar_respuesta(pregunta)

                    st.markdown(respuesta)

                    with st.expander("Consultar fuentes utilizadas"):
                        for fuente in fuentes:
                            st.markdown(f"**{fuente['archivo']}**")
                            st.caption(
                                f"Similitud: {fuente['score']:.4f}"
                            )
                            st.write(fuente["texto"][:700])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": respuesta,
                        "fuentes": fuentes
                    })

                except Exception as e:
                    st.error(f"Error al generar la respuesta: {e}")

    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 8. MÓDULO DE EXPLORACIÓN DOCUMENTAL
# =========================================================

with tab_documentos:

    st.subheader("Documentos indexados en Qdrant")

    try:
        documentos = explorar_documentos()

        if documentos:

            df_documentos = pd.DataFrame(documentos)

            archivos = sorted(df_documentos["archivo"].unique())

            st.metric("Fragmentos recuperados de la colección", len(documentos))
            st.metric("Documentos identificados", len(archivos))

            archivo_seleccionado = st.selectbox(
                "Seleccionar documento",
                archivos
            )

            filtrados = df_documentos[
                df_documentos["archivo"] == archivo_seleccionado
            ]

            st.write(f"Fragmentos del documento: {len(filtrados)}")

            for i, fila in enumerate(filtrados.itertuples(), 1):

                with st.expander(f"Fragmento {i}"):
                    st.write(fila.texto)

        else:
            st.info("No se encontraron fragmentos.")

    except Exception as e:
        st.error(f"Error explorando Qdrant: {e}")


# =========================================================
# 9. MÓDULO DE PREDICCIÓN ACADÉMICA (MACHINE LEARNING)
# =========================================================

with tab_ml:

    st.subheader("Estimación del rendimiento académico")

    st.write(
        "Este módulo utiliza un modelo MLP entrenado con "
        "el conjunto Students Performance in Exams."
    )

    modelo_ml = cargar_modelo_ml()

    if modelo_ml is None:

        st.warning(
            "El modelo MLP todavía no está disponible. "
            "Cuando exportemos el modelo entrenado como "
            "modelo_mlp.pkl, podremos habilitar las predicciones."
        )

    else:

        st.info(
            "La estimación es orientativa y no debe utilizarse "
            "como único criterio para tomar decisiones académicas."
        )

        with st.form("formulario_prediccion"):

            gender = st.selectbox(
                "Género",
                ["female", "male"]
            )

            race = st.selectbox(
                "Grupo étnico",
                ["group A", "group B", "group C", "group D", "group E"]
            )

            parental_education = st.selectbox(
                "Nivel educativo de los padres",
                [
                    "some high school",
                    "high school",
                    "some college",
                    "associate's degree",
                    "bachelor's degree",
                    "master's degree"
                ]
            )

            lunch = st.selectbox(
                "Tipo de alimentación",
                ["standard", "free/reduced"]
            )

            preparation = st.selectbox(
                "Preparación para el examen",
                ["none", "completed"]
            )

            ejecutar = st.form_submit_button(
                "Estimar rendimiento académico"
            )

        if ejecutar:

            datos = pd.DataFrame([{
                "gender": gender,
                "race/ethnicity": race,
                "parental level of education": parental_education,
                "lunch": lunch,
                "test preparation course": preparation
            }])

            try:
                prediccion = modelo_ml.predict(datos)[0]

                st.metric(
                    "Promedio académico estimado",
                    f"{prediccion:.2f} / 100"
                )

            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")


# =========================================================
# 10. INFORMACIÓN DEL SISTEMA
# =========================================================

with st.sidebar:

    st.title("Información del sistema")

    st.success("Qdrant conectado")
    st.success("Gemini conectado")

    st.write(f"**Modelo LLM:** {GEMINI_MODEL}")
    st.write(f"**Embeddings:** {EMBEDDING_MODEL}")
    st.write(f"**Colección:** {COLLECTION_NAME}")

    st.divider()

    st.caption(
        "Proyecto ACA 2 – Computación en la Nube\n\n"
        "Especialización en Inteligencia Artificial – CUN"
    )
