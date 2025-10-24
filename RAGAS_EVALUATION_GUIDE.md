# Guía de Evaluación con RAGAS

## 🎯 Script de Evaluación: `evaluate_with_ragas.py`

Este script evalúa tu sistema RAG usando las métricas de RAGAS:

### Métricas Evaluadas

#### ✅ Sin Ground Truth (Respuestas Correctas):
1. **Faithfulness (Fidelidad)** - ¿La respuesta es fiel al contexto recuperado?
2. **Answer Relevance (Relevancia)** - ¿La respuesta es relevante a la pregunta?

#### ✅ Con Ground Truth:
3. **Context Precision (Precisión)** - ¿Los contextos recuperados son precisos?
4. **Context Recall (Exhaustividad)** - ¿Se recuperó todo el contexto necesario?

---

## 🚀 Uso Rápido

### 1. Evaluación Básica (Sin Ground Truth)
```bash
python evaluate_with_ragas.py
```
- Usa 5 preguntas de ejemplo predefinidas
- Solo evalúa Faithfulness y Answer Relevance
- Genera respuestas automáticamente con tu RAG

### 2. Evaluación con Archivo de Preguntas
```bash
python evaluate_with_ragas.py --questions test_questions_simple.json
```

### 3. Evaluación Completa (Con Ground Truth)
```bash
python evaluate_with_ragas.py --questions test_questions_with_ground_truth.json --with-ground-truth
```
- Evalúa las 4 métricas principales
- Requiere archivo con respuestas correctas

### 4. Evaluar Materia Específica
```bash
python evaluate_with_ragas.py --materia "Probabilidad y estadística"
```

### 5. Personalizar Número de Chunks
```bash
python evaluate_with_ragas.py -k 10
```
- Recupera 10 chunks por pregunta (default: 5)

---

## 📁 Archivos de Entrada

### Formato JSON de Preguntas

**Sin ground truth:**
```json
[
    {
        "question": "¿Qué es una variable aleatoria?"
    },
    {
        "question": "¿Cuál es la definición de esperanza matemática?"
    }
]
```

**Con ground truth:**
```json
[
    {
        "question": "¿Qué es una variable aleatoria?",
        "ground_truth": "Una variable aleatoria es una función que asigna un valor numérico a cada resultado..."
    },
    {
        "question": "¿Cuál es la definición de esperanza matemática?",
        "ground_truth": "La esperanza matemática es el promedio ponderado..."
    }
]
```

---

## 📊 Archivos Generados

Todos los resultados se guardan en `./evaluation_results/`:

1. **`ragas_evaluation_results.json`** - Resultados detallados en JSON
2. **`ragas_evaluation_report.txt`** - Reporte legible en texto
3. **`test_data.json`** - Preguntas, respuestas y contextos usados

---

## 📈 Interpretación de Métricas

### Faithfulness (Fidelidad): 0.0 - 1.0
- **1.0** = La respuesta está completamente basada en el contexto
- **0.8-0.9** = Muy buena fidelidad
- **0.6-0.8** = Fidelidad aceptable
- **< 0.6** = La respuesta incluye información no presente en el contexto

**¿Qué mide?** Si el sistema "inventa" información o se mantiene fiel al contexto recuperado.

### Answer Relevance (Relevancia): 0.0 - 1.0
- **1.0** = La respuesta es perfectamente relevante a la pregunta
- **0.8-0.9** = Muy relevante
- **0.6-0.8** = Relevancia aceptable
- **< 0.6** = La respuesta no responde bien la pregunta

**¿Qué mide?** Si el sistema responde exactamente lo que se preguntó.

### Context Precision (Precisión): 0.0 - 1.0
⚠️ Requiere ground_truth

- **1.0** = Todos los chunks recuperados son relevantes
- **0.8-0.9** = La mayoría de chunks son relevantes
- **0.6-0.8** = Algunos chunks irrelevantes
- **< 0.6** = Muchos chunks no relevantes

**¿Qué mide?** La calidad del retrieval - si traes chunks útiles o "ruido".

### Context Recall (Exhaustividad): 0.0 - 1.0
⚠️ Requiere ground_truth

- **1.0** = Se recuperó toda la información necesaria
- **0.8-0.9** = Se recuperó casi toda la información
- **0.6-0.8** = Falta información relevante
- **< 0.6** = Falta mucha información necesaria

**¿Qué mide?** Si el retrieval encuentra TODO lo necesario para responder.

---

## 💡 Ejemplos Completos

### Ejemplo 1: Evaluación Rápida
```bash
# Evaluar con preguntas predefinidas
python evaluate_with_ragas.py

# Salida esperada:
# ✓ Faithfulness: 0.85 (BUENO)
# ✓ Answer Relevance: 0.78 (BUENO)
```

### Ejemplo 2: Evaluación Completa
```bash
# Usar archivo con ground truths
python evaluate_with_ragas.py \
    --questions test_questions_with_ground_truth.json \
    --with-ground-truth

# Salida esperada:
# ✓ Faithfulness: 0.85
# ✓ Answer Relevance: 0.78
# ✓ Context Precision: 0.82
# ✓ Context Recall: 0.75
```

### Ejemplo 3: Evaluar Solo Probabilidad
```bash
python evaluate_with_ragas.py \
    --materia "Probabilidad y estadística" \
    --questions test_questions_with_ground_truth.json \
    --with-ground-truth \
    -k 10
```

---

## 🔧 Opciones Avanzadas

```bash
python evaluate_with_ragas.py \
    --questions <archivo.json>        # Archivo con preguntas
    --with-ground-truth                # Usar ground_truths del archivo
    --materia <nombre>                 # Filtrar por materia
    -k <número>                        # Chunks a recuperar (default: 5)
    --output-dir <directorio>          # Dónde guardar resultados
```

---

## ⚙️ Qué Hace el Script Internamente

1. **Inicializa el Pipeline RAG**
   - Conecta a ChromaDB
   - Verifica que hay documentos cargados

2. **Para cada pregunta:**
   - Recupera K chunks relevantes del vector store
   - Genera una respuesta usando el LLM + contexto
   - Guarda pregunta, respuesta y contextos

3. **Evalúa con RAGAS:**
   - Calcula métricas según datos disponibles
   - Genera estadísticas (promedio, desviación, min, max)

4. **Guarda Resultados:**
   - JSON con resultados detallados
   - Reporte en texto plano
   - Datos de prueba usados

---

## 🎯 Valores Objetivo

Para un buen sistema RAG académico:

| Métrica | Objetivo Mínimo | Objetivo Ideal |
|---------|----------------|----------------|
| Faithfulness | > 0.75 | > 0.85 |
| Answer Relevance | > 0.70 | > 0.80 |
| Context Precision | > 0.70 | > 0.85 |
| Context Recall | > 0.65 | > 0.80 |

---

## 🐛 Troubleshooting

### Error: "No documents in database"
**Solución:**
```bash
python initialize_chroma.py
```

### Error: "ground_truth required"
**Problema:** Intentas evaluar Context Precision/Recall sin ground_truths

**Solución:**
- Agrega ground_truths a tu archivo JSON, O
- No uses `--with-ground-truth` (solo evaluará Faithfulness y Relevance)

### Métricas muy bajas (< 0.5)
**Posibles causas:**
- Chunks muy pequeños (aumenta CHUNK_SIZE)
- Estrategia de chunking inadecuada
- Embeddings de baja calidad
- K muy bajo (aumenta -k)

**Soluciones:**
1. Aumentar CHUNK_SIZE en .env
2. Probar estrategia 'semantic' o 'latex'
3. Aumentar K a 10-15 chunks
4. Verificar calidad de documentos fuente

### Evaluación muy lenta
**Normal:** RAGAS usa LLM para evaluar, puede tomar tiempo

**Para acelerar:**
- Reduce número de preguntas
- Reduce K (menos chunks por pregunta)
- Usa menos métricas

---

## 📚 Archivos de Ejemplo Incluidos

1. **`test_questions_simple.json`**
   - 5 preguntas básicas sin ground_truth
   - Para evaluación rápida

2. **`test_questions_with_ground_truth.json`**
   - 8 preguntas con respuestas correctas
   - Para evaluación completa

---

## 🎓 Mejores Prácticas

### 1. Empezar Simple
```bash
# Primera evaluación
python evaluate_with_ragas.py
```

### 2. Iterar con Ground Truth
```bash
# Crear archivo con tus preguntas y respuestas correctas
python evaluate_with_ragas.py \
    --questions mis_preguntas.json \
    --with-ground-truth
```

### 3. Optimizar Parámetros
- Si Context Precision es baja → Mejorar calidad de embeddings/chunking
- Si Context Recall es baja → Aumentar K
- Si Faithfulness es baja → Mejorar prompt del LLM
- Si Answer Relevance es baja → Verificar calidad de retrieval

### 4. Evaluar Regularmente
- Después de cambiar CHUNK_SIZE
- Después de cambiar estrategia de chunking
- Después de agregar nuevos documentos
- Después de cambiar modelo de embeddings

---

## 📝 Crear tus Propias Preguntas de Prueba

### Template Básico:
```json
[
    {
        "question": "Tu pregunta aquí",
        "ground_truth": "Respuesta correcta esperada"
    }
]
```

### Tips:
- Usa preguntas representativas de tu dominio
- Incluye diferentes niveles de dificultad
- Cubre diferentes temas/materias
- Las respuestas ground_truth deben ser precisas
- Mínimo 10-20 preguntas para evaluación robusta

---

## 🚀 Próximos Pasos

1. **Ejecuta tu primera evaluación:**
   ```bash
   python evaluate_with_ragas.py
   ```

2. **Revisa los resultados** en `./evaluation_results/`

3. **Analiza las métricas** y identifica áreas de mejora

4. **Optimiza tu sistema** basándote en los resultados

5. **Re-evalúa** después de cada cambio

---

¿Necesitas ayuda? Revisa los ejemplos o ejecuta:
```bash
python evaluate_with_ragas.py --help
```

