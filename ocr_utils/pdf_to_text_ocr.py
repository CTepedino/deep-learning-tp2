"""
Script para extraer texto de PDFs usando OCR (pytesseract).
Incluye extracción de texto de imágenes dentro del PDF.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
from typing import List, Tuple, Optional


def extract_text_from_pdf_with_ocr(pdf_path: str, 
                                 output_path: Optional[str] = None,
                                 languages: str = 'spa+eng',
                                 dpi: int = 300) -> str:
    """
    Extrae texto de un PDF usando OCR, incluyendo texto de imágenes.
    Combina texto directo y OCR de manera fluida, manteniendo el orden del documento.
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_path (str, optional): Ruta donde guardar el texto extraído
        languages (str): Idiomas para OCR (formato tesseract: 'spa+eng')
        dpi (int): DPI para renderizar las páginas del PDF
    
    Returns:
        str: Texto extraído del PDF
    """
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    print(f"Procesando PDF: {pdf_path}")
    print(f"Idiomas OCR: {languages}")
    print(f"DPI: {dpi}")
    
    # Abrir el PDF
    pdf_document = fitz.open(pdf_path)
    all_text = []
    
    # Procesar cada página
    for page_num in range(len(pdf_document)):
        print(f"Procesando página {page_num + 1}/{len(pdf_document)}")
        
        page = pdf_document[page_num]
        
        # 1. Extraer texto directo del PDF (si existe)
        page_text = page.get_text()
        
        # 2. Extraer texto de imágenes usando OCR
        # Renderizar la página como imagen
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 es el DPI por defecto
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convertir a PIL Image
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Aplicar OCR con pytesseract
        try:
            ocr_text = pytesseract.image_to_string(pil_image, lang=languages)
        except Exception as e:
            print(f"Error en OCR página {page_num + 1}: {e}")
            ocr_text = ""
        
        # Combinar texto directo y OCR de manera inteligente
        page_content = combine_text_and_ocr(page_text, ocr_text, page_num + 1)
        
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


def combine_text_and_ocr(direct_text: str, ocr_text: str, page_num: int) -> str:
    """
    Combina texto directo del PDF con texto extraído por OCR de manera inteligente.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        ocr_text (str): Texto extraído por OCR
        page_num (int): Número de página
    
    Returns:
        str: Texto combinado
    """
    
    # Limpiar y normalizar textos
    direct_text = direct_text.strip()
    ocr_text = ocr_text.strip()
    
    # Si solo hay texto directo, usarlo
    if direct_text and not ocr_text:
        return direct_text
    
    # Si solo hay texto OCR, usarlo
    if ocr_text and not direct_text:
        return ocr_text
    
    # Si hay ambos, combinar inteligentemente
    if direct_text and ocr_text:
        # Calcular similitud para decidir si combinar o usar el mejor
        similarity = calculate_text_similarity(direct_text, ocr_text)
        
        if similarity > 0.8:  # Muy similares, usar el más largo
            return direct_text if len(direct_text) > len(ocr_text) else ocr_text
        else:  # Diferentes, combinar
            # Usar texto directo como base y agregar contenido OCR que no esté presente
            combined = direct_text
            
            # Buscar párrafos en OCR que no estén en texto directo
            ocr_paragraphs = [p.strip() for p in ocr_text.split('\n') if p.strip()]
            direct_paragraphs = [p.strip() for p in direct_text.split('\n') if p.strip()]
            
            for ocr_para in ocr_paragraphs:
                if not any(calculate_text_similarity(ocr_para, direct_para) > 0.7 
                          for direct_para in direct_paragraphs):
                    combined += f"\n{ocr_para}"
            
            return combined
    
    return ""


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calcula similitud entre dos textos usando palabras comunes.
    
    Args:
        text1 (str): Primer texto
        text2 (str): Segundo texto
    
    Returns:
        float: Similitud entre 0 y 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalizar textos
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calcular intersección
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def extract_text_continuous_flow(pdf_path: str, 
                                output_path: Optional[str] = None,
                                languages: str = 'spa+eng',
                                dpi: int = 300) -> str:
    """
    Extrae texto de un PDF manteniendo el flujo continuo del documento.
    Combina texto directo y OCR de manera natural, sin separaciones artificiales.
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_path (str, optional): Ruta donde guardar el texto extraído
        languages (str): Idiomas para OCR (formato tesseract: 'spa+eng')
        dpi (int): DPI para renderizar las páginas del PDF
    
    Returns:
        str: Texto extraído del PDF en flujo continuo
    """
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    print(f"Procesando PDF en flujo continuo: {pdf_path}")
    print(f"Idiomas OCR: {languages}")
    print(f"DPI: {dpi}")
    
    # Abrir el PDF
    pdf_document = fitz.open(pdf_path)
    all_text = []
    
    # Procesar cada página
    for page_num in range(len(pdf_document)):
        print(f"Procesando página {page_num + 1}/{len(pdf_document)}")
        
        page = pdf_document[page_num]
        
        # 1. Extraer texto directo del PDF
        page_text = page.get_text()
        
        # 2. Extraer texto de imágenes usando OCR
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Aplicar OCR con pytesseract
        try:
            ocr_text = pytesseract.image_to_string(pil_image, lang=languages)
        except Exception as e:
            print(f"Error en OCR página {page_num + 1}: {e}")
            ocr_text = ""
        
        # Combinar de manera fluida
        page_content = create_continuous_text(page_text, ocr_text)
        
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


def create_continuous_text(direct_text: str, ocr_text: str) -> str:
    """
    Crea texto continuo combinando texto directo y OCR de manera natural.
    
    Args:
        direct_text (str): Texto extraído directamente del PDF
        ocr_text (str): Texto extraído por OCR
    
    Returns:
        str: Texto combinado en flujo continuo
    """
    
    direct_text = direct_text.strip()
    ocr_text = ocr_text.strip()
    
    # Si solo hay texto directo, usarlo
    if direct_text and not ocr_text:
        return direct_text
    
    # Si solo hay texto OCR, usarlo
    if ocr_text and not direct_text:
        return ocr_text
    
    # Si hay ambos, crear flujo continuo
    if direct_text and ocr_text:
        # Usar texto directo como base
        result = direct_text
        
        # Agregar contenido OCR que no esté duplicado
        ocr_lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        direct_lines = [line.strip() for line in direct_text.split('\n') if line.strip()]
        
        for ocr_line in ocr_lines:
            # Verificar si esta línea ya existe en el texto directo
            if not any(is_similar_line(ocr_line, direct_line) for direct_line in direct_lines):
                # Agregar al final del texto
                if result and not result.endswith((' ', '\n', '\t')):
                    result += " "
                result += ocr_line
        
        return result
    
    return ""


def is_similar_line(line1: str, line2: str, threshold: float = 0.8) -> bool:
    """
    Verifica si dos líneas son similares.
    
    Args:
        line1 (str): Primera línea
        line2 (str): Segunda línea
        threshold (float): Umbral de similitud
    
    Returns:
        bool: True si son similares
    """
    if not line1 or not line2:
        return False
    
    similarity = calculate_text_similarity(line1, line2)
    return similarity >= threshold


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


def extract_text_from_images_in_pdf(pdf_path: str, 
                                   output_path: Optional[str] = None,
                                   languages: str = 'spa+eng',
                                   dpi: int = 300) -> List[Tuple[int, str]]:
    """
    Extrae texto solo de las imágenes dentro del PDF.
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_path (str, optional): Ruta donde guardar los resultados
        languages (str): Idiomas para OCR
        dpi (int): DPI para renderizar las páginas
    
    Returns:
        List[Tuple[int, str]]: Lista de tuplas (número_página, texto_extraído)
    """
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
    
    print(f"Extrayendo texto de imágenes en PDF: {pdf_path}")
    
    pdf_document = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(pdf_document)):
        print(f"Procesando página {page_num + 1}/{len(pdf_document)}")
        
        page = pdf_document[page_num]
        
        # Renderizar página como imagen
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convertir a PIL Image
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Aplicar OCR con pytesseract
        try:
            ocr_text = pytesseract.image_to_string(pil_image, lang=languages)
        except Exception as e:
            print(f"Error en OCR página {page_num + 1}: {e}")
            ocr_text = ""
        
        if ocr_text.strip():
            results.append((page_num + 1, ocr_text))
            print(f"  - Texto extraído de página {page_num + 1}: {len(ocr_text)} caracteres")
    
    pdf_document.close()
    
    # Guardar resultados si se especifica output_path
    if output_path and results:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_num, text in results:
                f.write(f"=== PÁGINA {page_num} ===\n")
                f.write(text)
                f.write("\n\n")
        print(f"Resultados guardados en: {output_path}")
    
    return results


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
    
    parser = argparse.ArgumentParser(description='Extraer texto de PDFs usando OCR')
    parser.add_argument('pdf_path', help='Ruta al archivo PDF')
    parser.add_argument('-o', '--output', help='Archivo de salida para el texto')
    parser.add_argument('-l', '--languages', default='spa+eng', 
                       help='Idiomas para OCR (default: spa+eng)')
    parser.add_argument('-d', '--dpi', type=int, default=300,
                       help='DPI para renderizar páginas (default: 300)')
    parser.add_argument('--images-only', action='store_true',
                       help='Extraer solo texto de imágenes (OCR)')
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
        
        if args.images_only:
            results = extract_text_from_images_in_pdf(
                args.pdf_path, 
                args.output, 
                args.languages, 
                args.dpi
            )
            print(f"Extraído texto de {len(results)} páginas")
        elif args.continuous:
            text = extract_text_continuous_flow(
                args.pdf_path, 
                args.output, 
                args.languages, 
                args.dpi
            )
            print(f"Texto extraído en flujo continuo: {len(text)} caracteres")
        else:
            text = extract_text_from_pdf_with_ocr(
                args.pdf_path, 
                args.output, 
                args.languages, 
                args.dpi
            )
            print(f"Texto extraído: {len(text)} caracteres")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
