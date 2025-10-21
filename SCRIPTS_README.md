# 🎓 Scripts del Sistema RAG

Este documento explica cómo usar los diferentes scripts del sistema de generación de ejercicios académicos.

## 📋 Scripts Disponibles

### 1. 🗄️ `initialize_chroma.py` - Inicializar ChromaDB
**Propósito:** Poblar ChromaDB con documentos académicos

```bash
# Inicializar ChromaDB (solo si está vacío)
python initialize_chroma.py

# Forzar recarga de todos los documentos
python initialize_chroma.py --force
```

**¿Cuándo usarlo?**
- Primera vez que usas el sistema
- Quieres actualizar los documentos
- ChromaDB está vacío o corrupto

### 2. 🔍 `check_chroma.py` - Verificar ChromaDB
**Propósito:** Verificar el estado de ChromaDB

```bash
python check_chroma.py
```

**¿Cuándo usarlo?**
- Verificar si ChromaDB está poblado
- Diagnosticar problemas
- Antes de generar ejercicios

### 3. ⚙️ `generate_from_config.py` - Generar Ejercicios
**Propósito:** Generar ejercicios desde archivo de configuración

```bash
python generate_from_config.py config_example.json
```

**¿Cuándo usarlo?**
- Generar ejercicios personalizados
- Usar configuraciones específicas
- Automatizar la generación

### 4. 🧪 `test_rag.py` - Test Completo
**Propósito:** Probar todo el sistema end-to-end

```bash
python test_rag.py
```

**¿Cuándo usarlo?**
- Probar el sistema completo
- Generar ejercicios con parámetros fijos
- Desarrollo y debugging

## 🚀 Flujo de Trabajo Recomendado

### **Primera vez:**
```bash
# 1. Verificar estado
python check_chroma.py

# 2. Si está vacío, inicializar
python initialize_chroma.py

# 3. Generar ejercicios
python generate_from_config.py config_example.json
```

### **Uso diario:**
```bash
# 1. Verificar que todo esté bien
python check_chroma.py

# 2. Generar ejercicios
python generate_from_config.py config_example.json
```

### **Actualizar documentos:**
```bash
# 1. Recargar todos los documentos
python initialize_chroma.py --force

# 2. Verificar que se cargaron bien
python check_chroma.py

# 3. Generar ejercicios
python generate_from_config.py config_example.json
```

## 📝 Archivo de Configuración

Crea un archivo JSON con tus parámetros:

```json
{
  "materia": "Probabilidad y estadística",
  "unidad": "Variables Aleatorias", 
  "cantidad": 3,
  "nivel_dificultad": "intermedio",
  "tipo_ejercicio": "multiple_choice",
  "formato": "pdf"
}
```

### **Opciones disponibles:**

| Parámetro | Opciones | Descripción |
|-----------|----------|-------------|
| `materia` | "Probabilidad y estadística", "Sistemas de Inteligencia Artificial" | Materia académica |
| `unidad` | Cualquier texto | Unidad temática |
| `cantidad` | 1-10 | Número de ejercicios |
| `nivel_dificultad` | "basico", "intermedio", "avanzado" | Dificultad |
| `tipo_ejercicio` | "multiple_choice", "desarrollo", "practico", "teorico" | Tipo de ejercicio |
| `formato` | "txt", "pdf", "tex" | Formato de salida |

## 📁 Archivos Generados

Cada ejecución crea una carpeta con timestamp:
```
output/YYYYMMDD_HHMMSS/
├── Materia_completo.pdf      # Todo el contenido
├── Materia_ejercicio.pdf     # Solo preguntas
├── Materia_pistas.pdf        # Solo pistas
└── Materia_soluciones.pdf    # Solo respuestas
```

## ⚠️ Solución de Problemas

### **ChromaDB vacío:**
```bash
python check_chroma.py  # Verificar estado
python initialize_chroma.py  # Poblar
```

### **Errores de documentos:**
```bash
python initialize_chroma.py --force  # Recargar todo
```

### **No se generan ejercicios:**
```bash
python check_chroma.py  # Verificar ChromaDB
python generate_from_config.py config_example.json  # Probar generación
```

## 🎯 Resumen de Comandos

| Acción | Comando |
|--------|---------|
| Verificar ChromaDB | `python check_chroma.py` |
| Inicializar ChromaDB | `python initialize_chroma.py` |
| Generar ejercicios | `python generate_from_config.py config.json` |
| Test completo | `python test_rag.py` |
