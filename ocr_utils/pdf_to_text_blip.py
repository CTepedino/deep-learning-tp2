"""
Script para extraer texto de PDFs usando BLIP directamente.
Genera captions incondicionales de imágenes usando BlipProcessor y BlipForConditionalGeneration.
"""

import fitz  # PyMuPDF
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
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
    Extrae texto de un PDF usando BLIP directamente.
    
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
    
    # Cargar modelo BLIP
    print("Cargando modelo BLIP...")
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        print("✅ Modelo BLIP cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo BLIP: {e}")
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
        # Renderizar la página como imagen
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 es el DPI por defecto
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convertir a PIL Image
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Generar caption incondicional con BLIP
        try:
            caption = generate_unconditional_caption(pil_image, processor, model, device)
        except Exception as e:
            print(f"Error generando caption página {page_num + 1}: {e}")
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


def generate_unconditional_caption(image: Image.Image, processor, model, device: str) -> str:
    """
    Genera un caption incondicional para una imagen usando BLIP.
    
    Args:
        image (Image.Image): Imagen PIL
        processor: BlipProcessor
        model: BlipForConditionalGeneration
        device (str): Dispositivo a usar
    
    Returns:
        str: Caption generado
    """
    
    try:
        # Asegurar que la imagen esté en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generar caption incondicional
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(**inputs)
        
        # Decodificar el resultado
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
        
    except Exception as e:
        print(f"Error generando caption: {e}")
        return ""


def combine_text_and_caption(direct_text: str, caption: str, page_num: int) -> str:
    """
    Combina texto directo del PDF con caption generado.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        caption (str): Caption generado
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
        return f"[Descripción de la página: {caption}]"
    
    # Si hay ambos, combinar inteligentemente
    if direct_text and caption:
        # Usar texto directo como base
        combined = direct_text
        
        # Agregar caption si es relevante y no está duplicado
        if caption and not is_similar_content(direct_text, caption):
            combined += f"\n\n[Descripción adicional: {caption}]"
        
        return combined
    
    return ""


def is_similar_content(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """
    Verifica si dos textos son similares.
    
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
    Extrae texto de un PDF en flujo continuo usando BLIP.
    
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
    
    # Cargar modelo BLIP
    print("Cargando modelo BLIP...")
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        print("✅ Modelo BLIP cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo BLIP: {e}")
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
        
        # Generar caption incondicional con BLIP
        try:
            caption = generate_unconditional_caption(pil_image, processor, model, device)
        except Exception as e:
            print(f"Error generando caption página {page_num + 1}: {e}")
            caption = ""
        
        # Combinar de manera fluida
        page_content = create_continuous_text_captions(page_text, caption)
        
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


def create_continuous_text_captions(direct_text: str, caption: str) -> str:
    """
    Crea texto continuo combinando texto directo y caption de manera natural.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        caption (str): Caption generado
    
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
        return f"[Descripción: {caption}]"
    
    # Si hay ambos, crear flujo continuo
    if direct_text and caption:
        # Usar texto directo como base
        result = direct_text
        
        # Agregar caption si es relevante y no está duplicado
        if caption and not is_similar_content(direct_text, caption):
            # Agregar al final del texto
            if result and not result.endswith((' ', '\n', '\t')):
                result += " "
            result += f"[Descripción adicional: {caption}]"
        
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
    
    # Reemplazar múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text)
    
    # Reemplazar múltiples saltos de línea con uno solo
    text = re.sub(r'\n+', '\n', text)
    
    # Limpiar espacios al inicio y final de líneas
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # Reconstruir texto
    return '\n'.join(lines)


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


def main():
    """
    Función principal para probar el script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraer texto de PDFs usando BLIP')
    parser.add_argument('pdf_path', help='Ruta al archivo PDF')
    parser.add_argument('-o', '--output', help='Archivo de salida para el texto')
    parser.add_argument('-m', '--model', default='Salesforce/blip-image-captioning-base',
                       help='Modelo BLIP para image captioning')
    parser.add_argument('-d', '--dpi', type=int, default=300,
                       help='DPI para renderizar páginas (default: 300)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Dispositivo a usar (default: auto)')
    parser.add_argument('--continuous', action='store_true',
                       help='Extraer texto en flujo continuo (recomendado)')
    parser.add_argument('--info', action='store_true',
                       help='Mostrar información del PDF')
    
    args = parser.parse_args()
    
    try:
        if args.info:
            info = get_pdf_info(args.pdf_path)
            print("Información del PDF:")
            print(f"  Páginas: {info['num_pages']}")
            print(f"  Tamaño: {info['file_size']} bytes")
            print(f"  Metadatos: {info['metadata']}")
            return
        
        if args.continuous:
            text = extract_text_continuous_flow_blip(
                args.pdf_path, 
                args.output, 
                args.model,
                args.dpi,
                args.device
            )
            print(f"Texto extraído en flujo continuo: {len(text)} caracteres")
        else:
            text = extract_text_from_pdf_with_blip(
                args.pdf_path, 
                args.output, 
                args.model,
                args.dpi,
                args.device
            )
            print(f"Texto extraído: {len(text)} caracteres")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
