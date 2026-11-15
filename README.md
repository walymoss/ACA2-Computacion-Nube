
# Asistente de Permanencia Educativa

Sistema de inteligencia artificial que integra Retrieval-Augmented Generation (RAG) y Machine Learning para apoyar la consulta documental y el análisis del rendimiento académico.

## Arquitectura

- **Interfaz:** Streamlit.
- **Base vectorial:** Qdrant Cloud.
- **Modelo de lenguaje:** Google Gemini API.
- **Embeddings:** paraphrase-multilingual-MiniLM-L12-v2.
- **Machine Learning:** modelo MLP para estimar el rendimiento académico.

## Funcionalidades

1. Chatbot de consulta sobre permanencia estudiantil.
2. Recuperación semántica de documentos.
3. Respuestas fundamentadas con referencias documentales.
4. Exploración de documentos y fragmentos indexados.
5. Módulo de estimación del rendimiento académico.

## Estructura

```text
rag-permanencia-educativa/
├── app.py
├── requirements.txt
├── README.md
└── modelo_mlp.pkl
```

## Configuración

Configurar las siguientes variables en Streamlit Secrets:

- QDRANT_URL
- QDRANT_API_KEY
- GEMINI_API_KEY

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Proyecto académico

ACA 2 - Computación en la Nube  
Especialización en Inteligencia Artificial  
Corporación Unificada Nacional de Educación Superior (CUN)
