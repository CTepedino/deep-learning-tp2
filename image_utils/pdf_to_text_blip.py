"""
Script para extraer texto de PDFs usando BLIP directamente.
Genera captions incondicionales de imágenes (y las traduce al español) 
usando BlipProcessor, BlipForConditionalGeneration y un pipeline de traducción.
"""

import fitz  # PyMuPDF
import torch
# IMPORTACIÓN CLAVE: Se añade 'pipeline' para la traducción
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import io
import os
import time
from typing import Optional


def extract_text_from_pdf_with_blip(pdf_path: str, 
                                      output_path: Optional[str] = None,
                                      model_name: str = "Salesforce/blip-image-captioning-base",
                                      dpi: int = 300,
                                      device: str = "auto") -> str:
    """
    Extrae texto de un PDF usando BLIP directamente. (MODO PÁGINA A PÁGINA)
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_path (str, optional): Ruta donde guardar el texto extraído
        model_name (str): Modelo BLIP para image captioning
        dpi (int): DPI para renderizar las páginas del PDF
        device (str): Dispositivo a usar ("auto", "cpu", "cuda")
    
    Returns:
        str: Texto extraído del PDF
    """
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    print(f"Procesando PDF con BLIP: {pdf_path}")
    print(f"Modelo: {model_name}")
    print(f"DPI: {dpi}")
    print(f"Dispositivo: {device}")
    
    # Configurar dispositivo
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo BLIP y Traductor
    print("Cargando modelo BLIP...")
    translator = None
    try:
        # Arreglo 1: Usar use_fast=True para el procesador (corrige advertencia)
        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        
        # Arreglo 2: Cargar modelo (sin force_download, ya que el caché está limpio/fijo)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        print("✅ Modelo BLIP cargado exitosamente")

        # NUEVO: Cargar modelo de traducción
        print("Cargando modelo de traducción (en-es)...")
        translator = pipeline("translation_en_to_es", 
                              model="Helsinki-NLP/opus-mt-en-es",
                              device=device) # Usar GPU si está disponible
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
        
        # 2. Generar caption incondicional de la página usando BLIP
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 es el DPI por defecto
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convertir a PIL Image
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Generar caption incondicional con BLIP (y traducir)
        try:
            # NUEVO: Pasar el 'translator'
            caption = generate_unconditional_caption(pil_image, processor, model, device, translator)
        except Exception as e:
            print(f"  > Error generando caption página {page_num + 1}: {e}")
            caption = ""
        
        # Combinar texto directo y caption de manera inteligente
        page_content = combine_text_and_caption(page_text, caption, page_num + 1)
        
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


def generate_unconditional_caption(image: Image.Image, processor, model, device: str, translator_pipeline) -> str:
    """
    Genera un caption incondicional para una imagen usando BLIP y lo traduce al español.
    
    Args:
        image (Image.Image): Imagen PIL
        processor: BlipProcessor
        model: BlipForConditionalGeneration
        device (str): Dispositivo a usar
        translator_pipeline: Pipeline de traducción de Hugging Face (o None)
    
    Returns:
        str: Caption generado en ESPAÑOL (o inglés con prefijo si falla)
    """
    
    try:
        # Asegurar que la imagen esté en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # --- 1. Generar caption en INGLÉS ---
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # max_length más corto para captions más concisos
            out = model.generate(**inputs, max_length=50) 
        
        # Decodificar el resultado en INGLÉS
        caption_en = processor.decode(out[0], skip_special_tokens=True).strip()
        
        # --- 2. Traducir a ESPAÑOL ---
        if translator_pipeline and caption_en:
            try:
                # El traductor devuelve una lista, tomamos el primer elemento
                translation = translator_pipeline(caption_en, max_length=100) # Añadir max_length aquí también
                caption_es = translation[0]['translation_text']
                # print(f"  > EN: '{caption_en}' -> ES: '{caption_es}'") # Descomentar para debug
                return caption_es.strip()
            except Exception as e:
                print(f"  > Error traduciendo '{caption_en}': {e}")
                return f"[EN] {caption_en}" # Devolver en inglés si falla la traducción
        else:
            return f"[EN] {caption_en}" # Devolver en inglés si no hay traductor
            
    except Exception as e:
        print(f"  > Error generando caption: {e}")
        return ""


def combine_text_and_caption(direct_text: str, caption: str, page_num: int) -> str:
    """
    Combina texto directo del PDF con caption generado.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        caption (str): Caption generado (ya en español)
        page_num (int): Número de página
    
    Returns:
        str: Texto combinado
    """
    
    # Limpiar y normalizar textos
    direct_text = direct_text.strip()
    caption = caption.strip()
    
    # Si solo hay texto directo, usarlo
    if direct_text and not caption:
        return direct_text
    
    # Si solo hay caption, usarlo
    if caption and not direct_text:
        return f"[Descripción de la página {page_num}: {caption}]" # Añadido número de pág
    
    # Si hay ambos, combinar inteligentemente
    if direct_text and caption:
        # Usar texto directo como base
        combined = direct_text
        
        # Agregar caption si es relevante y no está duplicado
        # (La similaridad ahora compara texto en español con caption en español)
        if caption and not is_similar_content(direct_text, caption):
            combined += f"\n\n[Descripción de la imagen: {caption}]"
        
        return combined
    
    return ""


def is_similar_content(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """
    Verifica si dos textos son similares (Jaccard similarity).
    Umbral bajo (0.5) porque el caption puede ser muy diferente al texto.
    
    Args:
        text1 (str): Primer texto
        text2 (str): Segundo texto
        threshold (float): Umbral de similitud
    
    Returns:
        bool: True si son similares
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


def extract_text_continuous_flow_blip(pdf_path: str, 
                                          output_path: Optional[str] = None,
                                          model_name: str = "Salesforce/blip-image-captioning-base",
                                          dpi: int = 300,
                                          device: str = "auto") -> str:
    """
    Extrae texto de un PDF en flujo continuo usando BLIP (y traduce a español).
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_path (str, optional): Ruta donde guardar el texto extraído
        model_name (str): Modelo BLIP para image captioning
        dpi (int): DPI para renderizar las páginas del PDF
        device (str): Dispositivo a usar
    
    Returns:
        str: Texto extraído del PDF en flujo continuo
    """
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    print(f"Procesando PDF en flujo continuo con BLIP: {pdf_path}")
    print(f"Modelo: {model_name}")
    print(f"DPI: {dpi}")
    print(f"Dispositivo: {device}")
    
    # Configurar dispositivo
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo BLIP y Traductor
    print("Cargando modelo BLIP...")
    translator = None
    try:
        # Arreglo 1: Usar use_fast=True para el procesador (corrige advertencia)
        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        
        # Arreglo 2: Cargar modelo
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        print("✅ Modelo BLIP cargado exitosamente")

        # NUEVO: Cargar modelo de traducción
        print("Cargando modelo de traducción (en-es)...")
        translator = pipeline("translation_en_to_es", 
                              model="Helsinki-NLP/opus-mt-en-es",
                              device=device) # Usar GPU si está disponible
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
        
        # 1. Extraer texto directo del PDF
        page_text = page.get_text()
        
        # 2. Generar caption incondicional de la página usando BLIP
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Generar caption incondicional con BLIP (y traducir)
        try:
            # NUEVO: Pasar el 'translator'
            caption = generate_unconditional_caption(pil_image, processor, model, device, translator)
        except Exception as e:
            print(f"  > Error generando caption página {page_num + 1}: {e}")
            caption = ""
        
        # Combinar de manera fluida
        page_content = create_continuous_text_captions(page_text, caption, page_num + 1)
        
        if page_content.strip():
            all_text.append(page_content)
            # Solo agregar separador si no termina en punto
            if not page_content.strip().endswith(('.', '!', '?', ':')):
                all_text.append(" ")
            else:
                all_text.append(" ")
    
    pdf_document.close()
    
    # Combinar todo el texto en flujo continuo
    full_text = "".join(all_text)
    
    # Limpiar espacios múltiples y normalizar
    full_text = normalize_text_flow(full_text)
    
    # Guardar si se especifica output_path
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Texto guardado en: {output_path}")
    
    return full_text


def create_continuous_text_captions(direct_text: str, caption: str, page_num: int) -> str:
    """
    Crea texto continuo combinando texto directo y caption de manera natural.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        caption (str): Caption generado (ya en español)
        page_num (int): Número de página
    
    Returns:
        str: Texto combinado en flujo continuo
    """
    
    direct_text = direct_text.strip()
    caption = caption.strip()
    
    # Si solo hay texto directo, usarlo
    if direct_text and not caption:
        return direct_text
    
    # Si solo hay caption, usarlo
    if caption and not direct_text:
        return f"[Descripción (Pág. {page_num}): {caption}]"
    
    # Si hay ambos, crear flujo continuo
    if direct_text and caption:
        # Usar texto directo como base
        result = direct_text
        
        # Agregar caption si es relevante y no está duplicado
        if caption and not is_similar_content(direct_text, caption):
            # Agregar al final del texto
            if result and not result.endswith((' ', '\n', '\t')):
                result += " "
            result += f"[Descripción adicional (Pág. {page_num}): {caption}]"
        
        return result
    
    return ""


def normalize_text_flow(text: str) -> str:
    """
    Normaliza el flujo de texto eliminando espacios múltiples y mejorando la continuidad.
    
    Args:
        text (str): Texto a normalizar
    
    Returns:
        str: Texto normalizado
    """
    import re
    
    # Reemplazar múltiples saltos de línea y tabulaciones con un espacio
    text = re.sub(r'[\n\t]+', ' ', text)
    # Reemplazar múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_pdf_info(pdf_path: str) -> dict:
    """
    Obtiene información básica del PDF.
    
    Args:
        pdf_path (str): Ruta al archivo PDF
    
    Returns:
        dict: Información del PDF
    """
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
            import pytesseract
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


def combine_description_and_ocr(direct_text: str, description: str, ocr_text: str, page_num: int) -> str:
    """
    Combina texto directo, descripción BLIP y transcripción OCR en el formato solicitado.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        description (str): Descripción generada por BLIP (ya en español)
        ocr_text (str): Texto extraído por OCR
        page_num (int): Número de página
    
    Returns:
        str: Texto combinado en el formato [Descripción: ... | Transcripción: ...]
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
            result += f" | Transcripción de la imagen: {ocr_text}"
        
        result += "]"
    elif ocr_text and not is_similar_content(direct_text, ocr_text):
        # Solo transcripción si no hay descripción
        result += f"\n\n[Transcripción de la imagen: {ocr_text}]"
    
    return result


def main():
    """
    Función principal para probar el script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraer texto de PDFs usando BLIP + OCR')
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
    parser.add_argument('--continuous', action='store_true',
                        help='Extraer texto en flujo continuo (recomendado)')
    parser.add_argument('--unified', action='store_true',
                        help='Usar función unificada BLIP + OCR')
    parser.add_argument('--info', action='store_true',
                        help='Mostrar información del PDF')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        if args.info:
            info = get_pdf_info(args.pdf_path)
            print("Información del PDF:")
            print(f"  Páginas: {info['num_pages']}")
            print(f"  Tamaño: {info['file_size']} bytes")
            print(f"  Metadatos: {info['metadata']}")
            return
        
        if args.unified:
            text = extract_text_with_blip_and_ocr(
                args.pdf_path, 
                args.output, 
                args.model,
                args.languages,
                args.dpi,
                args.device
            )
            print(f"\nTexto extraído con BLIP + OCR: {len(text)} caracteres")
        elif args.continuous:
            text = extract_text_continuous_flow_blip(
                args.pdf_path, 
                args.output, 
                args.model,
                args.dpi,
                args.device
            )
            print(f"\nTexto extraído en flujo continuo: {len(text)} caracteres")
        else:
            text = extract_text_from_pdf_with_blip(
                args.pdf_path, 
                args.output, 
                args.model,
                args.dpi,
                args.device
            )
            print(f"\nTexto extraído: {len(text)} caracteres")
            
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        return 1
    
    end_time = time.time()
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
    return 0


if __name__ == "__main__":
    exit(main())