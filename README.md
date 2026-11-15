
# Asistente de Permanencia Educativa

Aplicación de consulta documental basada en Retrieval-Augmented Generation (RAG), desarrollada para el proyecto ACA 2 de Computación en la Nube.

## Arquitectura

La solución integra:

- **Qdrant Cloud:** almacenamiento y recuperación semántica de fragmentos documentales.
- **Sentence Transformers:** generación de embeddings con `paraphrase-multilingual-MiniLM-L12-v2`.
- **Google Gemini:** generación de respuestas mediante API.
- **Streamlit:** interfaz web interactiva.

## Funcionalidades

- Consultas en lenguaje natural sobre permanencia estudiantil.
- Recuperación semántica de documentos mediante Qdrant.
- Generación de respuestas fundamentadas en el contexto recuperado.
- Visualización de fuentes y fragmentos utilizados.
- Exploración de documentos indexados.
- Historial de conversación.

## Estructura del proyecto

```text
rag-permanencia-educativa/
├── app.py
├── requirements.txt
└── README.md
```

## Configuración

Configurar las siguientes variables en Streamlit Secrets:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `GEMINI_API_KEY`

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

## Tecnologías

Python, Streamlit, Qdrant Cloud, Sentence Transformers y Google Gemini.

## Proyecto académico

ACA 2 – Computación en la Nube  
Especialización en Inteligencia Artificial – CUN
