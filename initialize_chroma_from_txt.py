#!/usr/bin/env python3
"""
Script para poblar ChromaDB con documentos académicos desde archivos TXT
Este script carga documentos desde el directorio docstxt/ en lugar de docs/
Uso: python initialize_chroma_from_txt.py [--force]
"""

import argparse
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline

def main():
    parser = argparse.ArgumentParser(description='Inicializar ChromaDB con documentos académicos desde archivos TXT')
    parser.add_argument('--force', '-f', 
                       action='store_true',
                       help='Forzar recarga de todos los documentos (elimina datos existentes)')
    parser.add_argument('--directory', '-d',
                       default='./docstxt',
                       help='Directorio de documentos TXT (default: ./docstxt)')
    
    args = parser.parse_args()
    
    print("INICIALIZADOR DE CHROMADB (desde archivos TXT)")
    print("=" * 50)
    
    # Verificar que el directorio existe
    txt_dir = Path(args.directory)
    if not txt_dir.exists():
        print(f"ERROR: El directorio {args.directory} no existe")
        print("Asegúrate de haber convertido los PDFs a TXT primero")
        sys.exit(1)
    
    try:
        print("Creando pipeline RAG...")
        rag = create_rag_pipeline(reset_collection=args.force)
        
        if not args.force:
            print("Verificando contenido existente...")
            print("Usa --force para recargar todos los documentos")
        
        print(f"\nCargando documentos académicos desde TXT...")
        print(f"   Directorio: {args.directory}")
        print(f"   Formatos: TXT únicamente")
        print(f"   Saltando errores: Sí")
        
        load_result = rag.load_materials(
            data_directory=args.directory,
            file_extensions=[".txt"],
            skip_on_error=True
        )
        
        print("\n" + "=" * 50)
        print("RESULTADOS DE LA CARGA")
        print("=" * 50)
        
        if load_result.get('status') == 'success':
            print(f"✓ EXITO: {load_result.get('documents_loaded', 0)} documentos TXT cargados")
            print(f"✓ Chunks procesados: {load_result.get('chunks_created', 0)}")
            
            collection_info = load_result.get('collection_info', {})
            if collection_info:
                print(f"✓ Total documentos en DB: {collection_info.get('document_count', 0)}")
                print(f"✓ Vector store: {collection_info.get('persist_directory', 'N/A')}")
            
        elif load_result.get('status') in ['partial_error', 'warning']:
            print(f"⚠ CARGA PARCIAL: {load_result.get('message', 'Algunos documentos fallaron')}")
            print(f"✓ Documentos cargados: {load_result.get('documents_loaded', 0)}")
            print(f"✗ Documentos fallidos: {load_result.get('documents_failed', 0)}")
            print(f"✓ Chunks procesados: {load_result.get('chunks_created', 0)}")
            
        else:
            print(f"✗ ERROR: {load_result.get('message', 'Error desconocido')}")
            print(f"Documentos procesados: {load_result.get('documents_processed', 0)}")
            print(f"Chunks creados: {load_result.get('chunks_created', 0)}")
        
        if load_result.get('sources'):
            print(f"\nFUENTES TXT CARGADAS ({len(load_result.get('sources', []))}):")
            for i, source in enumerate(load_result.get('sources', [])[:10], 1):
                # Mostrar solo el nombre relativo del archivo
                source_path = Path(source)
                relative_path = source_path.relative_to(txt_dir) if txt_dir in source_path.parents else source_path.name
                print(f"   {i:2d}. {relative_path}")
            
            if len(load_result.get('sources', [])) > 10:
                print(f"   ... y {len(load_result.get('sources', [])) - 10} archivos más")
        
        if load_result.get('errors'):
            print(f"\n⚠ ERRORES ENCONTRADOS ({len(load_result.get('errors', []))}):")
            for i, error in enumerate(load_result.get('errors', [])[:5], 1):
                print(f"   {i}. {error}")
            if len(load_result.get('errors', [])) > 5:
                print(f"   ... y {len(load_result.get('errors', [])) - 5} errores más")
        
        print("\n" + "=" * 50)
        if load_result.get('status') == 'success':
            print("✓ ChromaDB inicializado exitosamente desde archivos TXT!")
            print("  Ahora puedes usar generate_from_config.py para crear ejercicios")
        elif load_result.get('status') in ['partial_error', 'warning']:
            print("⚠ ChromaDB inicializado parcialmente")
            print("  Algunos documentos no se pudieron cargar, pero puedes generar ejercicios")
        else:
            print("✗ Error inicializando ChromaDB")
            print("  Revisa los errores y vuelve a intentar")
        
       
        
    except KeyboardInterrupt:
        print("\n\n⚠ Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n✗ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

