#!/usr/bin/env python3
"""
Módulo para procesar archivos de docs y convertirlos a docstxt.
- Archivos .txt y .tex: se copian directamente
- Archivos .pdf: se procesan con pdf_to_text_unified y se guardan como .txt
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
import time

# Agregar el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from image_utils.pdf_to_text_unified import extract_text_with_blip_and_ocr


class DocsProcessor:
    """
    Procesador de archivos de docs a docstxt.
    """
    
    def __init__(self, docs_dir: str = "docs", docstxt_dir: str = "docstxt"):
        """
        Inicializa el procesador.
        
        Args:
            docs_dir (str): Directorio fuente (docs)
            docstxt_dir (str): Directorio destino (docstxt)
        """
        self.docs_dir = Path(docs_dir)
        self.docstxt_dir = Path(docstxt_dir)
        
        # Extensiones de texto que se copian directamente
        self.text_extensions = {'.txt', '.tex'}
        
        # Extensiones de PDF que se procesan
        self.pdf_extensions = {'.pdf'}
        
        # Extensiones soportadas
        self.supported_extensions = self.text_extensions | self.pdf_extensions
        
    def get_all_files(self) -> List[Tuple[Path, Path]]:
        """
        Obtiene todos los archivos soportados de docs y sus rutas destino.
        
        Returns:
            List[Tuple[Path, Path]]: Lista de tuplas (archivo_origen, archivo_destino)
        """
        files_to_process = []
        
        if not self.docs_dir.exists():
            print(f"❌ El directorio {self.docs_dir} no existe")
            return files_to_process
        
        # Recorrer recursivamente todos los archivos
        for file_path in self.docs_dir.rglob('*'):
            if file_path.is_file():
                # Verificar si es un archivo soportado
                if file_path.suffix.lower() in self.supported_extensions:
                    # Calcular ruta destino
                    relative_path = file_path.relative_to(self.docs_dir)
                    
                    # Para PDFs, cambiar extensión a .txt
                    if file_path.suffix.lower() == '.pdf':
                        dest_path = self.docstxt_dir / relative_path.with_suffix('.txt')
                    else:
                        dest_path = self.docstxt_dir / relative_path
                    
                    files_to_process.append((file_path, dest_path))
        
        return files_to_process
    
    def copy_text_file(self, source: Path, destination: Path) -> bool:
        """
        Copia un archivo de texto directamente.
        
        Args:
            source (Path): Archivo origen
            destination (Path): Archivo destino
            
        Returns:
            bool: True si se copió exitosamente
        """
        try:
            # Crear directorio padre si no existe
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivo
            shutil.copy2(source, destination)
            print(f"✅ Copiado: {source.name} -> {destination}")
            return True
            
        except Exception as e:
            print(f"❌ Error copiando {source.name}: {e}")
            return False
    
    def process_pdf_file(self, source: Path, destination: Path) -> bool:
        """
        Procesa un archivo PDF usando pdf_to_text_unified.
        
        Args:
            source (Path): Archivo PDF origen
            destination (Path): Archivo .txt destino
            
        Returns:
            bool: True si se procesó exitosamente
        """
        try:
            # Crear directorio padre si no existe
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"🔄 Procesando PDF: {source.name}")
            
            # Procesar PDF con BLIP + OCR
            extracted_text = extract_text_with_blip_and_ocr(
                pdf_path=str(source),
                output_path=str(destination),
                blip_model="Salesforce/blip-image-captioning-base",
                ocr_languages='spa+eng',
                dpi=300,
                device="auto"
            )
            
            if extracted_text.strip():
                print(f"✅ Procesado: {source.name} -> {destination.name} ({len(extracted_text)} caracteres)")
                return True
            else:
                print(f"⚠️  PDF procesado pero sin contenido: {source.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error procesando PDF {source.name}: {e}")
            return False
    
    def process_all_files(self, verbose: bool = True) -> dict:
        """
        Procesa todos los archivos de docs a docstxt.
        
        Args:
            verbose (bool): Mostrar información detallada
            
        Returns:
            dict: Estadísticas del procesamiento
        """
        print("🚀 Iniciando procesamiento de docs a docstxt...")
        print(f"📁 Origen: {self.docs_dir}")
        print(f"📁 Destino: {self.docstxt_dir}")
        
        # Obtener todos los archivos a procesar
        files_to_process = self.get_all_files()
        
        if not files_to_process:
            print("❌ No se encontraron archivos para procesar")
            return {"total": 0, "processed": 0, "errors": 0}
        
        print(f"📊 Archivos encontrados: {len(files_to_process)}")
        
        # Estadísticas
        stats = {
            "total": len(files_to_process),
            "processed": 0,
            "errors": 0,
            "text_files": 0,
            "pdf_files": 0,
            "text_errors": 0,
            "pdf_errors": 0
        }
        
        start_time = time.time()
        
        # Procesar cada archivo
        for i, (source, destination) in enumerate(files_to_process, 1):
            if verbose:
                print(f"\n[{i}/{len(files_to_process)}] Procesando: {source.relative_to(self.docs_dir)}")
            
            # Determinar tipo de archivo
            if source.suffix.lower() in self.text_extensions:
                stats["text_files"] += 1
                success = self.copy_text_file(source, destination)
                if success:
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    stats["text_errors"] += 1
                    
            elif source.suffix.lower() in self.pdf_extensions:
                stats["pdf_files"] += 1
                success = self.process_pdf_file(source, destination)
                if success:
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    stats["pdf_errors"] += 1
        
        # Mostrar estadísticas finales
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   Total archivos: {stats['total']}")
        print(f"   Procesados exitosamente: {stats['processed']}")
        print(f"   Errores: {stats['errors']}")
        print(f"   Archivos de texto: {stats['text_files']} (errores: {stats['text_errors']})")
        print(f"   Archivos PDF: {stats['pdf_files']} (errores: {stats['pdf_errors']})")
        print(f"   Tiempo total: {duration:.2f} segundos")
        
        if stats['errors'] > 0:
            print(f"\n⚠️  Se encontraron {stats['errors']} errores durante el procesamiento")
        else:
            print(f"\n✅ Todos los archivos se procesaron exitosamente")
        
        return stats
    
    def get_processing_summary(self) -> dict:
        """
        Obtiene un resumen de los archivos que serían procesados.
        
        Returns:
            dict: Resumen de archivos por tipo
        """
        files_to_process = self.get_all_files()
        
        summary = {
            "total_files": len(files_to_process),
            "text_files": 0,
            "pdf_files": 0,
            "by_extension": {},
            "by_directory": {}
        }
        
        for source, destination in files_to_process:
            ext = source.suffix.lower()
            summary["by_extension"][ext] = summary["by_extension"].get(ext, 0) + 1
            
            # Contar por directorio
            dir_name = source.parent.name
            summary["by_directory"][dir_name] = summary["by_directory"].get(dir_name, 0) + 1
            
            if ext in self.text_extensions:
                summary["text_files"] += 1
            elif ext in self.pdf_extensions:
                summary["pdf_files"] += 1
        
        return summary


def main():
    """
    Función principal para ejecutar el procesador desde línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Procesar archivos de docs a docstxt')
    parser.add_argument('--docs-dir', default='docs', help='Directorio origen (default: docs)')
    parser.add_argument('--docstxt-dir', default='docstxt', help='Directorio destino (default: docstxt)')
    parser.add_argument('--summary', action='store_true', help='Mostrar resumen sin procesar')
    parser.add_argument('--quiet', action='store_true', help='Modo silencioso')
    
    args = parser.parse_args()
    
    # Crear procesador
    processor = DocsProcessor(args.docs_dir, args.docstxt_dir)
    
    if args.summary:
        # Mostrar solo resumen
        summary = processor.get_processing_summary()
        print("📋 RESUMEN DE ARCHIVOS A PROCESAR:")
        print(f"   Total: {summary['total_files']}")
        print(f"   Archivos de texto: {summary['text_files']}")
        print(f"   Archivos PDF: {summary['pdf_files']}")
        print(f"   Por extensión: {summary['by_extension']}")
        print(f"   Por directorio: {summary['by_directory']}")
        return 0
    
    # Procesar archivos
    stats = processor.process_all_files(verbose=not args.quiet)
    
    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    exit(main())
