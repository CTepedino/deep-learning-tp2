#!/usr/bin/env python3
"""
Script unificado para extraer texto de PDFs usando BLIP + OCR.
Incluye todas las dependencias necesarias y manejo automático de instalaciones.
Formato de salida: [Descripción de la imagen: ... | Transcripción de la imagen: ...]
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional

def install_dependencies():
    """
    Instala las dependencias necesarias si no están disponibles.
    """
    required_packages = [
        'torch',
        'transformers',
        'pytesseract',
        'Pillow',
        'PyMuPDF',
        'sacremoses'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PyMuPDF':
                import fitz
            elif package == 'Pillow':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package} encontrado")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} no encontrado")
    
    if missing_packages:
        print(f"\n🔧 Instalando dependencias faltantes: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            try:
                print(f"Instalando {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} instalado exitosamente")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error instalando {package}: {e}")
                return False
    
    return True

def check_tesseract():
    """
    Verifica si Tesseract OCR está instalado en el sistema.
    """
    try:
        import pytesseract
        # Intentar obtener la versión de tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR encontrado (versión: {version})")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR no encontrado: {e}")
        print("📋 Para instalar Tesseract OCR:")
        print("   Windows: Descarga desde https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Linux: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")
        return False

def extract_text_with_blip_and_ocr(pdf_path: str, 
                                   output_path: Optional[str] = None,
                                   blip_model: str = "Salesforce/blip-image-captioning-base",
                                   ocr_languages: str = 'spa+eng',
                                   dpi: int = 300,
                                   device: str = "auto") -> str:
    """
    Extrae texto de un PDF combinando BLIP (descripción de imágenes) y OCR (transcripción de texto).
    Formato de salida: [Descripción de la imagen: ... | Transcripción de la imagen: ...]
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_path (str, optional): Ruta donde guardar el texto extraído
        blip_model (str): Modelo BLIP para image captioning
        ocr_languages (str): Idiomas para OCR (formato tesseract: 'spa+eng')
        dpi (int): DPI para renderizar las páginas del PDF
        device (str): Dispositivo a usar ("auto", "cpu", "cuda")
    
    Returns:
        str: Texto extraído del PDF con descripciones y transcripciones
    """
    
    # Importar dependencias
    import fitz  # PyMuPDF
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
    from PIL import Image
    import io
    import pytesseract
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    print(f"Procesando PDF con BLIP + OCR: {pdf_path}")
    print(f"Modelo BLIP: {blip_model}")
    print(f"Idiomas OCR: {ocr_languages}")
    print(f"DPI: {dpi}")
    print(f"Dispositivo: {device}")
    
    # Configurar dispositivo
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelos BLIP y Traductor
    print("Cargando modelo BLIP...")
    translator = None
    try:
        # Cargar BLIP
        processor = BlipProcessor.from_pretrained(blip_model, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(blip_model)
        model.to(device)
        print("✅ Modelo BLIP cargado exitosamente")

        # Cargar modelo de traducción
        print("Cargando modelo de traducción (en-es)...")
        translator = pipeline("translation_en_to_es", 
                              model="Helsinki-NLP/opus-mt-en-es",
                              device=device)
        print("✅ Modelo de traducción cargado.")
        
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return ""
    
    # Abrir el PDF
    pdf_document = fitz.open(pdf_path)
    all_text = []
    
    for page_num in range(len(pdf_document)):
        print(f"Procesando página {page_num + 1}/{len(pdf_document)}")
        
        page = pdf_document[page_num]
        
        # 1. Extraer texto directo del PDF (si existe)
        page_text = page.get_text()
        
        # 2. Renderizar página como imagen
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        # 3. Generar descripción con BLIP
        try:
            description = generate_unconditional_caption(pil_image, processor, model, device, translator)
        except Exception as e:
            print(f"  > Error generando descripción página {page_num + 1}: {e}")
            description = ""
        
        # 4. Extraer texto con OCR
        try:
            ocr_text = pytesseract.image_to_string(pil_image, lang=ocr_languages)
        except Exception as e:
            print(f"  > Error en OCR página {page_num + 1}: {e}")
            ocr_text = ""
        
        # 5. Combinar resultados en el formato solicitado
        page_content = combine_description_and_ocr(page_text, description, ocr_text, page_num + 1)
        
        if page_content.strip():
            all_text.append(page_content)
            all_text.append("\n\n")  # Separador entre páginas
    
    pdf_document.close()
    
    # Combinar todo el texto
    full_text = "".join(all_text)
    
    # Guardar si se especifica output_path
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Texto guardado en: {output_path}")
    
    return full_text


def generate_unconditional_caption(image, processor, model, device: str, translator_pipeline) -> str:
    """
    Genera un caption incondicional para una imagen usando BLIP y lo traduce al español.
    """
    import torch
    
    try:
        # Asegurar que la imagen esté en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generar caption en INGLÉS
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50) 
        
        # Decodificar el resultado en INGLÉS
        caption_en = processor.decode(out[0], skip_special_tokens=True).strip()
        
        # Traducir a ESPAÑOL
        if translator_pipeline and caption_en:
            try:
                translation = translator_pipeline(caption_en, max_length=100)
                caption_es = translation[0]['translation_text']
                return caption_es.strip()
            except Exception as e:
                print(f"  > Error traduciendo '{caption_en}': {e}")
                return f"[EN] {caption_en}"
        else:
            return f"[EN] {caption_en}"
            
    except Exception as e:
        print(f"  > Error generando caption: {e}")
        return ""


def combine_description_and_ocr(direct_text: str, description: str, ocr_text: str, page_num: int) -> str:
    """
    Combina texto directo, descripción BLIP y transcripción OCR en el formato solicitado.
    """
    # Limpiar y normalizar textos
    direct_text = direct_text.strip()
    description = description.strip()
    ocr_text = ocr_text.strip()
    
    # Si solo hay texto directo, usarlo
    if direct_text and not description and not ocr_text:
        return direct_text
    
    # Si no hay texto directo, crear formato combinado
    if not direct_text:
        # Crear formato: [Descripción: ... | Transcripción: ...]
        parts = []
        
        if description:
            parts.append(f"Descripción de la imagen: {description}")
        
        if ocr_text:
            parts.append(f"Transcripción de la imagen: {ocr_text}")
        
        if parts:
            return f"[{' | '.join(parts)}]"
        else:
            return ""
    
    # Si hay texto directo, combinarlo con descripción y OCR
    result = direct_text
    
    # Agregar descripción si es relevante
    if description and not is_similar_content(direct_text, description):
        result += f"\n\n[Descripción de la imagen: {description}"
        
        # Agregar transcripción si existe
        if ocr_text and not is_similar_content(direct_text, ocr_text):
            result += f" | Transcripción de la imagen: {ocr_text}]"
        
        result += "]"
    elif ocr_text and not is_similar_content(direct_text, ocr_text):
        # Solo transcripción si no hay descripción
        result += f"\n\n[Transcripción de la imagen: {ocr_text}]"
    
    return result


def is_similar_content(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """
    Verifica si dos textos son similares (Jaccard similarity).
    """
    if not text1 or not text2:
        return False
    
    # Normalizar textos
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return False
    
    # Calcular intersección
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    similarity = len(intersection) / len(union) if union else 0.0
    return similarity >= threshold


def get_pdf_info(pdf_path: str) -> dict:
    """
    Obtiene información básica del PDF.
    """
    import fitz
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    pdf_document = fitz.open(pdf_path)
    
    info = {
        'num_pages': len(pdf_document),
        'metadata': pdf_document.metadata,
        'file_size': os.path.getsize(pdf_path)
    }
    
    pdf_document.close()
    return info


def main():
    """
    Función principal para probar el script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraer texto de PDFs usando BLIP + OCR (con todas las dependencias)')
    parser.add_argument('pdf_path', help='Ruta al archivo PDF')
    parser.add_argument('-o', '--output', help='Archivo de salida para el texto')
    parser.add_argument('-m', '--model', default='Salesforce/blip-image-captioning-base',
                        help='Modelo BLIP para image captioning')
    parser.add_argument('-l', '--languages', default='spa+eng',
                        help='Idiomas para OCR (default: spa+eng)')
    parser.add_argument('-d', '--dpi', type=int, default=300,
                        help='DPI para renderizar páginas (default: 300)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Dispositivo a usar (default: auto)')
    parser.add_argument('--info', action='store_true',
                        help='Mostrar información del PDF')
    parser.add_argument('--install-deps', action='store_true',
                        help='Instalar dependencias automáticamente')
    
    args = parser.parse_args()
    
    # Verificar e instalar dependencias si es necesario
    if args.install_deps:
        print("🔧 Verificando e instalando dependencias...")
        if not install_dependencies():
            print("❌ Error instalando dependencias")
            return 1
        
        if not check_tesseract():
            print("⚠️  Tesseract OCR no está disponible - OCR no funcionará")
    
    start_time = time.time()
    
    try:
        if args.info:
            info = get_pdf_info(args.pdf_path)
            print("Información del PDF:")
            print(f"  Páginas: {info['num_pages']}")
            print(f"  Tamaño: {info['file_size']} bytes")
            print(f"  Metadatos: {info['metadata']}")
            return
        
        text = extract_text_with_blip_and_ocr(
            args.pdf_path, 
            args.output, 
            args.model,
            args.languages,
            args.dpi,
            args.device
        )
        print(f"\nTexto extraído con BLIP + OCR: {len(text)} caracteres")
            
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    end_time = time.time()
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
    return 0


if __name__ == "__main__":
    exit(main())
