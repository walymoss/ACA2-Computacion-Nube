
import os
import streamlit as st

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


# =========================================================
# 2. CREDENCIALES
# =========================================================

def get_secret(name):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


if not all([QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY]):
    st.error(
        "Faltan credenciales. Configura QDRANT_URL, "
        "QDRANT_API_KEY y GEMINI_API_KEY en los Secrets."
    )
    st.stop()


# =========================================================
# 3. CONEXIONES Y MODELOS
# =========================================================

@st.cache_resource
def conectar_qdrant(url, api_key):
    return QdrantClient(
        url=url,
        api_key=api_key
    )


@st.cache_resource
def cargar_embeddings():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def conectar_gemini(api_key):
    return genai.Client(api_key=api_key)


try:
    qdrant = conectar_qdrant(QDRANT_URL, QDRANT_API_KEY)
    embedding_model = cargar_embeddings()
    gemini = conectar_gemini(GEMINI_API_KEY)

    # Verificar que la colección exista
    qdrant.get_collection(COLLECTION_NAME)

except Exception as e:
    st.error(f"Error al inicializar los servicios: {e}")
    st.stop()


# =========================================================
# 4. RECUPERACIÓN SEMÁNTICA DESDE QDRANT
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
# 5. GENERACIÓN DE RESPUESTAS RAG CON GEMINI
# =========================================================

def generar_respuesta(pregunta):

    resultados = recuperar_contexto(
        pregunta,
        top_k=3
    )

    if not resultados:
        return "No se encontraron documentos relevantes.", []

    contexto = "\n\n".join([
        f"[Fuente: {r.payload.get('archivo', 'Documento sin nombre')}]\n"
        f"{r.payload.get('texto', '')[:1200]}"
        for r in resultados
        if r.payload
    ])

    if not contexto.strip():
        return "No se encontró contenido documental para responder.", []

    prompt = f"""
Eres un asistente especializado en permanencia estudiantil.

Instrucciones:
- Responde exclusivamente con base en el contexto proporcionado.
- No inventes información ni agregues datos externos.
- Cita los documentos utilizados en tu respuesta.
- Si el contexto no contiene información suficiente, indícalo claramente.
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

    respuesta = response.text or "No fue posible generar una respuesta."

    fuentes = [
        {
            "archivo": r.payload.get("archivo", "Sin nombre"),
            "texto": r.payload.get("texto", ""),
            "score": r.score
        }
        for r in resultados
        if r.payload
    ]

    return respuesta, fuentes


# =========================================================
# 6. EXPLORACIÓN DE LA BASE DOCUMENTAL
# =========================================================

@st.cache_data(ttl=300)
def explorar_documentos():

    documentos = []
    offset = None

    while True:

        puntos, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for punto in puntos:

            payload = punto.payload or {}

            documentos.append({
                "archivo": payload.get("archivo", "Sin nombre"),
                "texto": payload.get("texto", "")
            })

        if offset is None:
            break

    return documentos


# =========================================================
# 7. INTERFAZ PRINCIPAL
# =========================================================

st.title("🎓 Asistente de Permanencia Educativa")

st.write(
    "Asistente inteligente para consultar documentos sobre "
    "permanencia estudiantil mediante recuperación semántica "
    "y generación aumentada por recuperación (RAG)."
)

tab_chat, tab_documentos = st.tabs([
    "💬 Asistente RAG",
    "📚 Base documental"
])


# =========================================================
# 8. MÓDULO DE CHAT
# =========================================================

with tab_chat:

    st.subheader("Consulta académica")

    st.caption(
        "Realiza preguntas sobre permanencia, acompañamiento "
        "estudiantil y estrategias institucionales."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial de conversación
    for mensaje in st.session_state.messages:

        with st.chat_message(mensaje["role"]):

            st.markdown(mensaje["content"])

            if mensaje.get("fuentes"):

                with st.expander("Consultar fuentes utilizadas"):

                    for fuente in mensaje["fuentes"]:

                        st.markdown(f"**{fuente['archivo']}**")

                        if fuente["score"] is not None:
                            st.caption(
                                f"Similitud: {fuente['score']:.4f}"
                            )

                        st.write(fuente["texto"][:700])

    pregunta = st.chat_input("Escribe tu pregunta...")

    if pregunta:

        st.session_state.messages.append({
            "role": "user",
            "content": pregunta
        })

        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):

            with st.spinner(
                "Consultando documentos y generando respuesta..."
            ):

                try:

                    respuesta, fuentes = generar_respuesta(pregunta)

                    st.markdown(respuesta)

                    if fuentes:

                        with st.expander("Consultar fuentes utilizadas"):

                            for fuente in fuentes:

                                st.markdown(f"**{fuente['archivo']}**")

                                if fuente["score"] is not None:
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

                    st.error(
                        f"Error al generar la respuesta: {e}"
                    )

    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 9. MÓDULO DE EXPLORACIÓN DOCUMENTAL
# =========================================================

with tab_documentos:

    st.subheader("Documentos indexados en Qdrant")

    try:

        documentos = explorar_documentos()

        if documentos:

            archivos = sorted(set(
                documento["archivo"]
                for documento in documentos
            ))

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fragmentos indexados",
                    len(documentos)
                )

            with col2:
                st.metric(
                    "Documentos identificados",
                    len(archivos)
                )

            archivo_seleccionado = st.selectbox(
                "Seleccionar documento",
                archivos
            )

            filtrados = [
                documento
                for documento in documentos
                if documento["archivo"] == archivo_seleccionado
            ]

            st.write(
                f"Fragmentos del documento: {len(filtrados)}"
            )

            for i, fragmento in enumerate(filtrados, 1):

                with st.expander(f"Fragmento {i}"):

                    st.write(fragmento["texto"])

        else:

            st.info("No se encontraron fragmentos en la colección.")

    except Exception as e:

        st.error(
            f"Error explorando la base documental: {e}"
        )


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
