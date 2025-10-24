#Sistema RAG para Generación de Ejercicios Académicos

Un sistema inteligente que genera ejercicios académicos personalizados usando técnicas de RAG (Retrieval-Augmented Generation) y ChromaDB.

## ¿Qué hace este sistema?

Este sistema puede:
-  **Generar ejercicios** de materias específicas (Probabilidad y estadística, Sistemas de IA)
-  **Personalizar dificultad** (básico, intermedio, avanzado)
-  **Crear diferentes tipos** de ejercicios (múltiple choice, desarrollo, prácticos, teóricos)
-  **Exportar en múltiples formatos** (PDF, TXT, TEX)
-  **Buscar información** en documentos académicos
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

### Primera vez: Inicializar la base de datos

```bash
# Poblar la base de datos con documentos académicos
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

### Verificar el estado de chromaDB
```bash
python check_chroma.py
```

### Recargar documentos (si actualizas archivos)
```bash
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
├── 📁 docs/                        # Documentos académicos (PDFs)
├── 📁 docstxt/                     # Documentos procesados (TXT)
├── 📁 data/                        # Base de datos ChromaDB
├── 📁 output/                      # Ejercicios generados
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

