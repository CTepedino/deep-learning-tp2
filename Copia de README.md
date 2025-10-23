# Estructura de Documentos Académicos

## 📁 Organización de Carpetas

Este sistema utiliza una **estructura jerárquica de carpetas** para extraer metadata automáticamente sin necesidad de configuración manual.

### Estructura Recomendada

```
docs/
  └── [MATERIA]/
      └── Unidad_[NN]_[TEMA]/
          └── [TIPO]/
              └── archivos.pdf
```

### Ejemplo Completo

```
docs/
  ├── Probabilidad_y_estadistica/
  │   ├── Unidad_01_Variables_Aleatorias/
  │   │   ├── apuntes/
  │   │   │   ├── teoria_variables.pdf
  │   │   │   └── conceptos_basicos.pdf
  │   │   ├── ejercicios/
  │   │       ├── guia_practica_1.pdf
  │   │       └── problemas_resueltos.pdf
  │   │   
  │   ├── Unidad_02_Distribucion_Normal/
  │   │   ├── apuntes/
  │   │   │   └── distribucion_normal.pdf
  │   │   ├── ejercicios/
  │   │   │   └── ejercicios_distribucion.pdf
  │   │   └── guias/
  │   │       └── guia_normal.pdf
  │   └── Unidad_03_Regresion_Lineal/
  │       ├── apuntes/
  │       └── ejercicios/
  └── SIA/
      ├── Unidad_01_Clustering/
      │   ├── apuntes/
      │   │   └── kmeans_teoria.pdf
      │   └── ejercicios/
      │       └── clustering_practice.pdf
      ├── Unidad_02_Redes_Neuronales/
      │   ├── apuntes/
      │   └── practicas/
      └── Unidad_03_Machine_Learning_Supervisado/
          ├── apuntes/
          └── ejercicios/
```

## 🏗️ Niveles de la Jerarquía

### Nivel 1: Materia (Obligatorio)

Nombre de la materia. Usa **guiones bajos** para separar palabras.

**Ejemplos válidos:**
- `Probabilidad_y_estadistica`
- `SIA`
- `Algebra`
- `Analisis_Matematico`
- `Sistemas_Operativos`

**Metadata generada:**
```python
{
    'materia': 'Probabilidad y estadistica'  # Se convierten _ a espacios
}
```

### Nivel 2: Unidad/Tema (Recomendado)

Formato: `Unidad_[NN]_[TEMA]`

- `[NN]`: Número de unidad con ceros a la izquierda (01, 02, 03, ...)
- `[TEMA]`: Descripción del tema con guiones bajos

**Ejemplos válidos:**
- `Unidad_01_Variables_Aleatorias`
- `Unidad_02_Distribucion_Normal`
- `Unidad_03_Clustering`
- `Unidad_10_Redes_Neuronales`

**Metadata generada:**
```python
{
    'unidad_numero': 1,
    'unidad_tema': 'Variables Aleatorias',
    'unidad': 'Variables Aleatorias'
}
```

**Variantes aceptadas:**
- `Unidad01_Tema` (sin guion después de Unidad)
- `unidad_01_tema` (minúsculas)
- `Tema_01` (sin palabra "Unidad")

### Nivel 3: Tipo de Documento (Recomendado)

Categoría del documento. Usa nombres en plural y minúsculas.

**Tipos recomendados:**
- `apuntes/` - Apuntes teóricos
- `ejercicios/` - Ejercicios y problemas
- `guias/` - Guías de estudio o práctica
- `examenes/` - Exámenes completos
- `parciales/` - Exámenes parciales
- `finales/` - Exámenes finales
- `practicas/` - Trabajos prácticos

**Metadata generada:**
```python
{
    'tipo_documento': 'ejercicios'
}
```

## 📝 Convenciones de Nombres

### ✅ Buenas Prácticas

1. **Usa guiones bajos** en lugar de espacios
   - ✅ `Variables_Aleatorias`
   - ❌ `Variables Aleatorias`

2. **Numera las unidades** con ceros a la izquierda
   - ✅ `Unidad_01_`, `Unidad_02_`, ..., `Unidad_10_`
   - ❌ `Unidad_1_`, `Unidad_2_`, ..., `Unidad_10_`

3. **Usa nombres descriptivos** para las materias
   - ✅ `Probabilidad_y_estadistica`
   - ❌ `Proba`, `PE`

4. **Mantén consistencia** en los nombres de tipos
   - ✅ Siempre `ejercicios/` (plural)
   - ❌ Mezclar `ejercicio/` y `ejercicios/`

### 🎯 Nombres de Archivo (Opcional)

Puedes agregar información adicional en el nombre del archivo:

**Nivel de dificultad:**
- `ejercicios_basico.pdf` → metadata `nivel_sugerido: 'introductorio'`
- `problemas_avanzado.pdf` → metadata `nivel_sugerido: 'avanzado'`
- `guia_intermedio.pdf` → metadata `nivel_sugerido: 'intermedio'`

**Ejemplos de buenos nombres:**
- `teoria_distribucion_normal.pdf`
- `ejercicios_resueltos_probabilidad.pdf`
- `parcial_2023_recuperatorio.pdf`
- `guia_practica_1_basico.pdf`

## 🔍 Metadata Generada Automáticamente

### Ejemplo de Ruta

```
docs/Probabilidad_y_estadistica/Unidad_02_Distribucion_Normal/ejercicios/guia_practica_2.pdf
```

### Metadata Extraída

```python
{
    # Metadata básica
    'source': 'docs/Probabilidad_y_estadistica/Unidad_02_Distribucion_Normal/ejercicios/guia_practica_2.pdf',
    'filename': 'guia_practica_2.pdf',
    'file_type': '.pdf',
    'file_size': 245678,
    'parent_dir': 'ejercicios',
    
    # Metadata académica (extraída de carpetas)
    'materia': 'Probabilidad y estadistica',
    'unidad_numero': 2,
    'unidad_tema': 'Distribucion Normal',
    'unidad': 'Distribucion Normal',
    'tipo_documento': 'ejercicios',
    
    # Metadata del contenido (extraída del texto)
    'palabras_clave': ['distribución', 'normal', 'probabilidad']
}
```

## 🛡️ Fallbacks (Compatibilidad)

Si no sigues la estructura exacta, el sistema tiene **fallbacks inteligentes**:

### 1. Sin carpeta `docs/`
El sistema buscará la carpeta `data/` o usará los últimos niveles de la ruta.

### 2. Sin estructura de Unidad
```
docs/Probabilidad_y_estadistica/teoria_variables.pdf
```
- ✅ Se detectará la materia
- ⚠️ No se detectará la unidad
- ✅ Se intentará detectar el tipo por el nombre del archivo

### 3. Archivos sueltos
```
docs/apunte_probabilidad.pdf
```
- ⚠️ Materia se detectará por keywords en el nombre
- ⚠️ Tipo se detectará por keywords ("apunte" → tipo: "apuntes")

## 📊 Uso con el Sistema RAG

### Carga de Documentos

```python



# Ver metadata extraída
for doc in docs[:3]:
    print(f"Materia: {doc.metadata.get('materia')}")
    print(f"Unidad: {doc.metadata.get('unidad')}")
    print(f"Tipo: {doc.metadata.get('tipo_documento')}")
    print("-" * 50)
```

### Generación de Ejercicios con Filtros

```python
from src.rag_pipeline import create_rag_pipeline

rag = create_rag_pipeline()

# Cargar materiales
rag.load_materials(directory_path="./docs")

# Generar ejercicios usando metadata automática
result = rag.generate_exercises(
    query_params={
        "materia": "Probabilidad y estadistica",  # Filtrado automático
        "unidad": "Distribucion Normal",          # Filtrado por unidad
        "tipo_ejercicio": "multiple_choice",
        "cantidad": 3,
        "nivel_dificultad": "intermedio"
    },
    k_retrieval=5
)
```

## 🚀 Migración de Documentos Existentes

### Script de Ayuda

Si ya tienes documentos y quieres migrarlos a esta estructura:

```bash
# Crear estructura base
mkdir -p docs/Probabilidad_y_estadistica/Unidad_01_Variables_Aleatorias/{apuntes,ejercicios,examenes}
mkdir -p docs/SIA/Unidad_01_Clustering/{apuntes,ejercicios}

# Mover archivos existentes
mv data/raw/probabilidad/*.pdf docs/Probabilidad_y_estadistica/Unidad_01_Variables_Aleatorias/apuntes/
```

## ✅ Validación de Estructura

Puedes validar que tu estructura es correcta ejecutando:

```python
from src.utils.validate_docs import validate_docs_structure

# Validar estructura
result = validate_docs_structure("./docs")

if result['valid']:
    print("✅ Estructura válida")
    print(f"Materias encontradas: {result['stats']['materias']}")
    print(f"Total archivos: {result['stats']['archivos_totales']}")
else:
    print("❌ Problemas encontrados:")
    for issue in result['issues']:
        print(f"  - {issue}")
```

## 📚 Extensión para Nuevas Materias

Para agregar una nueva materia:

1. Crea la carpeta con el nombre de la materia:
   ```bash
   mkdir -p docs/Algebra
   ```

2. Crea las unidades:
   ```bash
   mkdir -p docs/Algebra/Unidad_01_Matrices
   mkdir -p docs/Algebra/Unidad_02_Determinantes
   ```

3. Crea las carpetas de tipos:
   ```bash
   mkdir -p docs/Algebra/Unidad_01_Matrices/{apuntes,ejercicios,examenes}
   ```

4. Coloca tus archivos en las carpetas correspondientes

5. ✅ La metadata se extraerá automáticamente

## 💡 Consejos

- **Mantén consistencia**: Decide un estilo y síguelo
- **Documenta excepciones**: Si un archivo no sigue la estructura, déjalo claro en el nombre
- **Usa Git**: Versiona tus documentos para trackear cambios
- **Backup regular**: Haz copias de seguridad de la carpeta `docs/`

---

**Última actualización:** Octubre 2025  
**Mantenido por:** Grupo 7 - Deep Learning TP2

