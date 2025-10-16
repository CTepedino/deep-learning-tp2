# Sistema RAG para Generación de Ejercicios Académicos

Sistema de Retrieval-Augmented Generation (RAG) diseñado para generar ejercicios académicos basados en materiales oficiales de la carrera de Ingeniería Informática del ITBA.

**MVP**: Soporte para **Probabilidad y estadística** y **Sistemas de Inteligencia Artificial (SIA)**.

## 🎯 Características

- **Carga flexible de documentos**: Soporte para PDF, TXT y TEX
- **Fragmentación inteligente**: Splitting optimizado para documentos académicos
- **Embeddings con LangChain**: Integración completa con ecosistema LangChain
- **Base de datos vectorial**: ChromaDB con persistencia automática
- **Búsqueda semántica**: Con filtros por materia, tipo y dificultad
- **Generación de ejercicios**: 4 tipos diferentes con OpenAI
- **Metadata académica**: Detección automática de materias (Probabilidad y estadística, SIA) y tipos de documento
- **Arquitectura estándar**: Usando LangChain para máxima flexibilidad

## 📚 Materias Soportadas (MVP)

- **Probabilidad y estadística**: Ejercicios matemáticos y simbólicos
- **Sistemas de Inteligencia Artificial (SIA)**: Ejercicios teóricos y conceptuales

## 📋 Tipos de Ejercicios Soportados

- **multiple_choice**: Preguntas de opción múltiple (A, B, C, D)
- **desarrollo**: Preguntas de desarrollo teórico
- **practico**: Problemas prácticos con cálculos
- **teorico**: Preguntas conceptuales teóricas

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd deep-learning-tp2
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
```bash
# Copiar archivo de ejemplo
cp env.example .env

# Editar .env con tus API keys
nano .env
```

Variables requeridas:
```env
OPENAI_API_KEY=tu_api_key_aqui
```

Variables opcionales:
```env
HF_TOKEN=tu_huggingface_token
EMBEDDING_PROVIDER=huggingface  # o "openai"
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📖 Uso Básico

### 1. Inicializar el sistema
```python
from src.rag_pipeline import create_rag_pipeline

# Crear pipeline RAG
rag_pipeline = create_rag_pipeline(
    collection_name="itba_ejercicios",
    embedding_model="all-MiniLM-L6-v2",  # Modelo por defecto
    reset_collection=False
)
```

### 2. Cargar materiales académicos
```python
# Cargar desde directorio
result = rag_pipeline.load_materials(
    directory_path="./data/raw",
    use_academic_metadata=True
)

print(f"Documentos cargados: {result['documents_loaded']}")
print(f"Chunks creados: {result['chunks_created']}")
```

### 3. Generar ejercicios
```python
# Parámetros de la consulta
query_params = {
    "materia": "Probabilidad y estadística",
    "unidad": "Distribución normal",
    "cantidad": 3,
    "nivel_dificultad": "intermedio",
    "tipo_ejercicio": "multiple_choice"
}

# Generar ejercicios
result = rag_pipeline.generate_exercises(
    query_params=query_params,
    tipo_ejercicio="multiple_choice",
    k_retrieval=5
)

# Mostrar resultados
for i, exercise in enumerate(result['ejercicios'], 1):
    print(f"Ejercicio {i}: {exercise['pregunta']}")
    print(f"Opciones: {exercise['opciones']}")
    print(f"Respuesta: {exercise['respuesta_correcta']}")
```

## 📁 Estructura del Proyecto

```
deep-learning-tp2/
├── src/                          # Código fuente
│   ├── data_loading.py          # Carga de documentos
│   ├── text_processing.py       # Fragmentación de texto
│   ├── vector_store.py          # ChromaDB con LangChain
│   ├── retriever.py             # Retriever con LangChain
│   ├── query_utils.py           # Construcción de consultas
│   ├── generator.py             # Generación con LLM
│   └── rag_pipeline.py          # Pipeline principal
├── data/
│   ├── raw/                     # Materiales originales
│   └── processed/               # ChromaDB persistente
├── notebooks/
│   └── rag_demo.ipynb          # Demo completo
├── requirements.txt             # Dependencias con LangChain
├── env.example                  # Template de variables
└── README.md                    # Este archivo
```

## 🔧 Configuración Avanzada

### Cambiar modelo de embeddings
```python
# Usar modelo diferente de sentence-transformers
rag_pipeline = create_rag_pipeline(
    embedding_model="all-mpnet-base-v2"  # Mayor calidad, más lento
)
```

### Modelos disponibles
- `all-MiniLM-L6-v2` (384 dims) - Rápido y eficiente (por defecto)
- `all-mpnet-base-v2` (768 dims) - Mayor calidad
- `multi-qa-MiniLM-L6-cos-v1` (384 dims) - Optimizado para Q&A

### Configurar generador
```python
# Ajustar creatividad
rag_pipeline.update_generator_settings(
    model_name="gpt-4o",     # Modelo más potente
    temperature=0.8          # Más creativo
)
```

## 📊 Ejemplo de Output

```json
{
  "ejercicios": [
    {
      "pregunta": "¿Cuál es la probabilidad de que una variable aleatoria normal estándar sea menor que 1.96?",
      "opciones": [
        "0.95",
        "0.975", 
        "0.025",
        "0.05"
      ],
      "respuesta_correcta": "B",
      "pista": "Usa la tabla de la distribución normal estándar",
      "solucion": "Para Z ~ N(0,1), P(Z < 1.96) = 0.975 según la tabla estándar."
    }
  ],
  "metadata": {
    "materia": "Probabilidad y estadística",
    "unidad": "Distribución normal",
    "tipo_ejercicio": "multiple_choice",
    "cantidad": 1,
    "nivel_dificultad": "intermedio",
    "chunks_recuperados": 5,
    "fuentes": ["apunte_probabilidad.pdf", "guia_ejercicios.pdf"],
    "modelo_usado": "gpt-4o-mini"
  }
}
```

## 🧪 Demo Interactivo

Ejecuta el notebook de demostración:
```bash
jupyter notebook notebooks/rag_demo.ipynb
```

El demo incluye:
- Configuración del sistema
- Carga de materiales de ejemplo
- Generación de diferentes tipos de ejercicios
- Análisis del sistema
- Ejemplos de uso avanzado

## 🔍 Búsqueda de Materiales

```python
# Búsqueda básica
results = rag_pipeline.search_materials(
    query="distribución normal",
    k=5
)

# Búsqueda con filtros
results = rag_pipeline.search_materials(
    query="algoritmos",
    k=3,
    filter_dict={"materia": "Sistemas de Inteligencia Artificial"}
)
```

## 📈 Métricas y Análisis

```python
# Obtener información del sistema
info = rag_pipeline.get_system_info()
print(f"Documentos en vector store: {info['vector_store']['document_count']}")
print(f"Modelo de embeddings: {info['embeddings']['model_name']}")
print(f"Dimensión de embeddings: {info['embeddings']['embedding_dimension']}")
```

## 🛠️ Desarrollo

### Agregar nuevos tipos de ejercicios
1. Editar `src/generator.py`
2. Agregar template en `_setup_prompts()`
3. Implementar función de validación
4. Actualizar documentación

### Agregar nuevos formatos de documento
1. Editar `src/data_loading.py`
2. Agregar loader en `DocumentLoader.__init__()`
3. Implementar extractor de metadata si es necesario

## 🐛 Solución de Problemas

### Error: "OPENAI_API_KEY no encontrada"
- Verifica que el archivo `.env` existe y contiene la API key
- Asegúrate de cargar las variables con `load_dotenv()`

### Error: "No se encontró contexto relevante"
- Verifica que los materiales están cargados en el vector store
- Ajusta los parámetros de búsqueda (`k_retrieval`)
- Revisa que la consulta sea específica

### Error: "ChromaDB no se puede conectar"
- Verifica permisos de escritura en `data/processed/`
- Elimina la colección existente si es necesario (`reset_collection=True`)

## 📝 Notas Importantes

- **Materiales académicos**: El sistema está optimizado para documentos del ITBA (Probabilidad y estadística, SIA)
- **LangChain Integration**: Usa el ecosistema estándar de LangChain para máxima flexibilidad
- **Persistencia**: ChromaDB guarda embeddings en disco para reutilización
- **Metadata**: El sistema detecta automáticamente materias y tipos de documento
- **Arquitectura estándar**: Compatible con el ecosistema completo de LangChain

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

Para preguntas o problemas:
- Abre un issue en GitHub
- Revisa la documentación en `notebooks/rag_demo.ipynb`
- Consulta los logs del sistema para debugging

---

**Desarrollado para el TP2 de Deep Learning - ITBA**
