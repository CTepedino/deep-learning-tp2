# OCR Utils - Extracción de Texto de PDFs

Este módulo proporciona herramientas para extraer texto de archivos PDF usando OCR (Reconocimiento Óptico de Caracteres), incluyendo texto de imágenes dentro del PDF.

**Nota**: Este módulo usa `pytesseract` que es más fácil de instalar en Windows que `tesserocr`.

## Características

- ✅ Extracción de texto directo del PDF
- ✅ OCR de imágenes dentro del PDF usando pytesseract
- ✅ Soporte para múltiples idiomas
- ✅ Configuración de DPI para mejor calidad
- ✅ Procesamiento por páginas
- ✅ Guardado automático de resultados

## Dependencias

```bash
pip install pytesseract PyMuPDF Pillow
```

### Instalación de Tesseract

**Windows:**
```bash
# Opcion 1: Instalacion automatica
python ocr_utils/install_tesseract_windows.py

# Opcion 2: Instalacion manual
# 1. Descarga desde: https://github.com/UB-Mannheim/tesseract/wiki
# 2. Instala seleccionando "Add to PATH"
# 3. Instala pytesseract
pip install pytesseract
```

**Linux/Ubuntu:**
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

**macOS:**
```bash
brew install tesseract
pip install pytesseract
```

## Uso

### Uso Básico (Flujo Continuo - RECOMENDADO)

```python
from ocr_utils.pdf_to_text_ocr import extract_text_continuous_flow

# Extraer texto en flujo continuo (mejor para documentos académicos)
text = extract_text_continuous_flow("documento.pdf", "output.txt")
print(f"Texto extraído: {len(text)} caracteres")
```

### Uso Tradicional

```python
from ocr_utils.pdf_to_text_ocr import extract_text_from_pdf_with_ocr

# Extraer todo el texto (con separaciones por página)
text = extract_text_from_pdf_with_ocr("documento.pdf", "output.txt")
print(f"Texto extraído: {len(text)} caracteres")
```

### Solo OCR de Imágenes

```python
from ocr_utils.pdf_to_text_ocr import extract_text_from_images_in_pdf

# Extraer solo texto de imágenes
results = extract_text_from_images_in_pdf("documento.pdf", "ocr_output.txt")
for page_num, text in results:
    print(f"Página {page_num}: {len(text)} caracteres")
```

### Configuración Avanzada

```python
# Con idiomas específicos y DPI personalizado
text = extract_text_from_pdf_with_ocr(
    "documento.pdf",
    "output.txt",
    languages="spa+eng",  # Español + Inglés
    dpi=600  # Mayor resolución para mejor OCR
)
```

### Línea de Comandos

```bash
# Extraer texto en flujo continuo (RECOMENDADO)
python ocr_utils/pdf_to_text_ocr.py documento.pdf --continuous -o output.txt

# Extraer todo el texto (tradicional)
python ocr_utils/pdf_to_text_ocr.py documento.pdf -o output.txt

# Solo OCR de imágenes
python ocr_utils/pdf_to_text_ocr.py documento.pdf --images-only -o ocr.txt

# Con idiomas específicos
python ocr_utils/pdf_to_text_ocr.py documento.pdf --continuous -l "spa+eng" -o output.txt

# Información del PDF
python ocr_utils/pdf_to_text_ocr.py documento.pdf --info
```

## Parámetros

### `extract_text_continuous_flow()` (RECOMENDADO)

- `pdf_path` (str): Ruta al archivo PDF
- `output_path` (str, opcional): Archivo donde guardar el texto
- `languages` (str): Idiomas para OCR (formato tesseract)
- `dpi` (int): DPI para renderizar páginas (default: 300)

**Características:**
- ✅ Combina texto directo y OCR de manera fluida
- ✅ Mantiene el flujo natural del documento
- ✅ Elimina duplicaciones automáticamente
- ✅ Ideal para documentos académicos

### `extract_text_from_pdf_with_ocr()`

- `pdf_path` (str): Ruta al archivo PDF
- `output_path` (str, opcional): Archivo donde guardar el texto
- `languages` (str): Idiomas para OCR (formato tesseract)
- `dpi` (int): DPI para renderizar páginas (default: 300)

### Idiomas Soportados

- `spa`: Español
- `eng`: Inglés
- `spa+eng`: Español + Inglés
- `fra`: Francés
- `deu`: Alemán
- Y muchos más...

## Ejemplos de Uso

### Integración con el Sistema RAG

```python
from ocr_utils.pdf_to_text_ocr import extract_text_continuous_flow
from src.rag_pipeline import create_rag_pipeline

# Extraer texto de PDF académico en flujo continuo
text = extract_text_continuous_flow("docs/apunte_probabilidad.pdf")

# Crear pipeline RAG
rag_pipeline = create_rag_pipeline()

# Procesar el texto extraído
# (el pipeline puede manejar texto directamente)
```

### Procesamiento de Documentos Académicos

```python
from ocr_utils.pdf_to_text_ocr import extract_text_continuous_flow

# Procesar apuntes con imágenes y texto mixto
def process_academic_document(pdf_path, output_path):
    """Procesa documentos académicos manteniendo el flujo natural."""
    text = extract_text_continuous_flow(
        pdf_path, 
        output_path,
        languages="spa+eng",  # Español + Inglés para documentos del ITBA
        dpi=600  # Alta resolución para mejor OCR
    )
    
    print(f"Documento procesado: {len(text)} caracteres")
    print(f"Guardado en: {output_path}")
    return text

# Usar
text = process_academic_document("docs/teorica_probabilidad.pdf", "output/teorica.txt")
```

### Procesamiento por Lotes

```python
import os
from ocr_utils.pdf_to_text_ocr import extract_text_from_pdf_with_ocr

def process_pdf_folder(input_folder, output_folder):
    """Procesar todos los PDFs en una carpeta."""
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.pdf', '.txt'))
            
            try:
                text = extract_text_from_pdf_with_ocr(input_path, output_path)
                print(f"Procesado: {filename} -> {len(text)} caracteres")
            except Exception as e:
                print(f"Error procesando {filename}: {e}")

# Usar
process_pdf_folder("docs/", "output/")
```

## Notas Importantes

1. **Calidad del OCR**: Depende de la calidad de las imágenes en el PDF
2. **DPI**: Valores más altos (600+) dan mejor calidad pero son más lentos
3. **Idiomas**: Especificar los idiomas correctos mejora la precisión
4. **Memoria**: PDFs grandes pueden requerir mucha memoria RAM

## Solución de Problemas

### Error: "tesseract not found"
```bash
# Instalar tesseract primero
# Windows: usar conda o descargar wheel
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### Error: "No module named pytesseract"
```bash
pip install pytesseract
```

### Error: "TesseractNotFoundError"
```bash
# Windows: Instalar Tesseract y agregar al PATH
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### Calidad de OCR baja
- Aumentar DPI (600+)
- Verificar idiomas correctos
- Preprocesar imágenes si es necesario
