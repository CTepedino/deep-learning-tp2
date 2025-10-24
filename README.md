#Sistema RAG para Generación de Ejercicios Académicos

Un sistema inteligente que genera ejercicios académicos personalizados usando técnicas de RAG (Retrieval-Augmented Generation) y ChromaDB.

## ¿Qué hace este sistema?

Este sistema puede:
-  **Procesar PDFs con imágenes** usando BLIP (descripciones semánticas) + OCR (extracción de texto)
-  **Generar ejercicios** de materias específicas (Probabilidad y estadística, Sistemas de IA)
-  **Personalizar dificultad** (básico, intermedio, avanzado)
-  **Crear diferentes tipos** de ejercicios (múltiple choice, desarrollo, prácticos, teóricos)
-  **Exportar en múltiples formatos** (PDF, TXT, TEX)
-  **Buscar información** en documentos académicos procesados
-  **Evaluar la calidad** de los ejercicios generados

##Instalación

### Verificar que tienes Python

Abre tu terminal (Terminal en Mac, CMD en Windows) y escribe:
```bash
python --version
```

Si no tienes Python, descárgalo desde [python.org](https://www.python.org/downloads/)

### Crear entorno virtual

```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Mac/Linux:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### Instalar dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar todas las dependencias
pip install -r requirements.txt
```

### Configurar variables de entorno

1. **Copia el archivo de ejemplo:**
```bash
cp env.example .env
```

2. **Edita el archivo `.env`** con tu editor favorito:

3. **Configura tu API Key de OpenAI:**
```env
OPENAI_API_KEY=tu_clave_de_openai_aqui
```

## Uso Básico

### Preprocesamiento de documentos (Primera vez)

**Importante:** Para usar OCR con pytesseract es necesario tener instalado el motor de Google Tesseract OCR. Pasos de instalación aquí: https://pypi.org/project/pytesseract/

```bash
# Procesar todos los PDFs de docs/ a docstxt/ (extrae texto de imágenes con BLIP + OCR)
python src/process_docs.py

# Ver qué archivos se procesarían sin procesarlos
python src/process_docs.py --summary
```

Este paso convierte:
- **PDFs con imágenes** → **Texto procesado** (usando BLIP para descripciones + OCR para texto)
- **Archivos .txt/.tex** → **Se copian directamente**

### Inicializar la base de datos

**Opción 1: Con procesamiento de imágenes (Recomendado)**
```bash
# Poblar la base de datos con documentos académicos procesados (desde docstxt/)
python initialize_chroma_from_txt.py
```

**Opción 2: Sin procesamiento de imágenes (Más rápido)**
```bash
# Poblar la base de datos ignorando imágenes de los PDFs (desde docs/)
python initialize_chroma.py
```


### Generar ejercicios

#### Opción 1: Usando archivo de configuración 
```bash
python generate_from_config.py config_example.json
```

#### Opción 2: Usando el generador de queries 
```bash
python query_generator.py
```

### Ver los resultados

Los ejercicios se guardan en la carpeta `output/` con un timestamp:
```
output/20250124_143022/
├── Probabilidad_y_estadística_completo.pdf --(Contiene ejercicio, pista y solucion)
├── Probabilidad_y_estadística_ejercicio.pdf
├── Probabilidad_y_estadística_pistas.pdf
└── Probabilidad_y_estadística_soluciones.pdf
```

## Configuración Avanzada

### Parámetros disponibles

| Parámetro | Opciones | Descripción |
|-----------|----------|-------------|
| `materia` | "Probabilidad y estadística", "Sistemas de Inteligencia Artificial" | Materia académica |
| `consulta_libre` | Cualquier texto | Descripción específica de lo que quieres |
| `cantidad` | 1-7 | Número de ejercicios |
| `nivel_dificultad` | "basico", "intermedio", "avanzado" | Dificultad |
| `tipo_ejercicio` | "multiple_choice", "desarrollo", "practico", "teorico" | Tipo de ejercicio |
| `formato` | "txt", "pdf", "tex" | Formato de salida |



## Comandos Útiles

### Procesamiento de documentos
```bash
# Procesar PDFs con BLIP + OCR
python src/process_docs.py

# Ver resumen de archivos a procesar
python src/process_docs.py --summary

# Procesar con directorios personalizados
python src/process_docs.py --docs-dir mi_docs --docstxt-dir mi_docstxt
```

### Gestión de base de datos
```bash
# Verificar el estado de chromaDB
python check_chroma.py
```

### Recargar documentos (si actualizas archivos)
```bash
# Recargar desde documentos procesados (con imágenes)
python initialize_chroma_from_txt.py --force

# O recargar desde documentos originales (ignora imágenes)
python initialize_chroma.py --force
```

### Muestra algunos chunks de la base de datos (util para ver formato y metadata)
```bash
python show_chunk.py
```
### Limpiar la base de datos
```bash
python clean_chroma.py
```

## Estructura del Proyecto

```
deep-learning-tp2/
├── 📄 README.md                    # Este archivo
├── 📄 requirements.txt             # Dependencias
├── 📄 config_example.json          # Configuración de ejemplo
├── 📄 env.example                  # Variables de entorno
├── 📁 src/                         # Código fuente
├── 📁 docs/                        # Documentos académicos (PDFs originales)
├── 📁 docstxt/                     # Documentos procesados (TXT con texto extraído)
├── 📁 data/                        # Base de datos ChromaDB
├── 📁 output/                      # Ejercicios generados
└── 📁 image_utils/                 # Utilidades para procesamiento de imágenes
```

## Evaluación del Sistema

### Evaluar con RAGAS
```bash
python evaluate_with_ragas.py --questions questions.json [--with-ground-truth]

Evaluar sin ground truth solo provee las metricas faithfulness y context precision

```

### Evaluar con LLM Judge
```bash
python evaluate_with_llm_judge.py --input questions.json
```

