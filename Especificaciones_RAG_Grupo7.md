# Proyecto RAG – Especificaciones Internas

## 1. Objetivos

Crear un sistema de **generación de ejercicios para alumnos** basado en RAG (Retrieval-Augmented Generation).  
El objetivo es que el modelo genere preguntas, problemas o ejercicios prácticos **ajustados al material oficial de las materias de la carrera de Ingeniería Informática del ITBA**, evitando incluir temas fuera del plan de estudios o con enfoques distintos a los de la institución.

El sistema debe:
- Recuperar información de los materiales oficiales (apuntes, programas, guías, exámenes viejos, bibliografía).
- Generar ejercicios de distintos tipos (multiple choice, desarrollo, prácticos, teóricos).
- Proveer soluciones y pistas.
- Adaptarse al **nivel de dificultad** y **tema** indicados por el usuario.

---

## 2. Alcance

### MVP inicial
- Limitar el dominio a 2 materias:
  - **Probabilidad y estadística** → ejercicios matemáticos y simbólicos (mayor complejidad de parsing).
  - **SIA (Sistemas de Inteligencia Artificial)** → ejercicios teóricos o conceptuales.
- Usar materiales disponibles del **campus del ITBA** y fuentes públicas internas (no web scraping externo).

### Inputs esperados
```json
{
  "materia": "Probabilidad y estadística",
  "unidad": "Distribución normal",
  "tipo_ejercicio": "multiple choice",
  "cantidad": 5,
  "nivel_dificultad": "intermedio"
}
```

### Outputs esperados
- Lista de ejercicios generados con estructura:
```json
{
  "pregunta": "...",
  "opciones": ["A", "B", "C", "D"],
  "respuesta_correcta": "C",
  "pista": "...",
  "solución": "..."
}
```

---

## 3. Requisitos Técnicos

### RAG Core
- Arquitectura **Transformer + Retrieval-Augmented Generation**.
- Módulos esperados:
  1. **Data Loading** (lectura de documentos `.txt`, `.pdf`, `.tex`).
  2. **Text Splitting** (fragmentación por tema/unidad).
  3. **Vector Database** (almacenamiento de embeddings de pasajes + metadata).
  4. **Retriever** (búsqueda de contexto relevante).
  5. **Generator** (modelo LLM / Transformer fine-tuned o API OpenAI).
  6. **Post-processing** (validación del formato del ejercicio, revisión de dificultad).

### Persistencia
- **Base de datos vectorial:** embeddings + metadata `{materia, unidad, palabras_clave}`.
- **DB estructurada opcional:** plan de estudios, correlatividades, plantillas de ejercicios.
- Se pueden combinar datos estructurados (plan, correlativas) y no estructurados (PDFs, apuntes, guías).

### Dataset
- Datos de **apuntes oficiales del ITBA**, **guías de ejercicios**, **exámenes viejos**.
- Formatos: `.pdf`, `.txt`, `.tex`.
- Origen: materiales personales o del campus (sin scraping web).

### Métricas
- Precisión y coherencia temática (ejercicio correcto y dentro del alcance).
- Variedad de ejercicios (no repeticiones).
- Calibración de dificultad (introductorio / parcial / avanzado).

---

## 4. Decisiones del Grupo

- Limitar el alcance a **la carrera de Informática (ITBA)**.
- Comenzar con **Probabilidad y estadística** y **SIA** para el MVP.
- Usar **API de OpenAI** para la parte generativa (profesoras proveen key).
- Dataset híbrido (estructurado + no estructurado).
- Usar **prompt chaining** y **reranking** como técnicas de *Advanced RAG*:
  - Paso 1: recuperar y filtrar los pasajes más relevantes.
  - Paso 2: generar preguntas a partir del contexto limpio.
- Evaluar la posibilidad de incluir **calibración automática de dificultad** a partir de embeddings o metadatos.

---

## 5. Criterios de Evaluación (según consigna)

**Ejercicio 1**
- Aplicación de técnicas de RAG (data loading, retrieval, embeddings, etc.).
- Implementación de preprocesamiento y postprocesamiento.
- Creación y uso de dataset relevante.
- Métricas claras para evaluar resultados.

**Ejercicio 2**
- Definir (no implementar):
  - Principios de Responsible AI / Safety.
  - Estrategias de despliegue a producción.

**Presentación (30 min)**
- Explicación del problema.
- Desafíos técnicos.
- Decisiones de diseño.
- Resultados.
- Conclusiones.

---

## 6. Entregables

| Etapa | Fecha | Entrega |
|-------|--------|----------|
| Pre-entrega | 19/09 – 17:59hs | Mail con descripción del proyecto, input/output, datos, DB, desafíos |
| Informe (Ej. 2) | 22/10 – 17:59hs | Documento 2 páginas con consideraciones de Responsible AI y deployment |
| Entrega final | 24/10 – 17:59hs | Repositorio con código, README.md, presentación y dataset |
| Exposición oral | Fecha aleatoria asignada por docentes | 30 min |

> Se penaliza 1 punto por cada entrega tardía.

---

## 7. Pendientes / TODOs

- [ ] Definir formato exacto del esquema de metadata por materia/unidad.
- [ ] Decidir herramienta para la base vectorial (`FAISS`, `Chroma`, `Pinecone`, etc).
- [ ] Probar extracción de texto desde `.tex` y `.pdf` de apuntes reales.
- [ ] Elegir métricas automáticas o semiautomáticas para evaluar calidad de ejercicios.
- [ ] Implementar pipeline mínimo con una materia antes del 19/09 (pre-entrega).
- [ ] Definir template de salida (JSON/Markdown) para ejercicios.
- [ ] Aclarar con docentes si se permite usar embeddings preentrenados externos.
- [ ] Agregar consideraciones de Responsible AI y estrategias de producción (Ej. 2).

---

## 8. Notas

- Se debe mencionar explícitamente que **no se usa material fuera del plan de estudios ITBA**.
- Puede incorporarse un sistema de tags por materia/unidad para evitar mezclar temas.
- Output idealmente en formato `.tex` o `.md` para facilitar revisión.
- Evaluar uso de prompt templates para tipos de ejercicios (e.g. “multiple choice”, “desarrollo”, “problema práctico”).
