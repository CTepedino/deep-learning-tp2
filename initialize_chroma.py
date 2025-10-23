#!/usr/bin/env python3
"""
Script para poblar ChromaDB con documentos académicos
Uso: python initialize_chroma.py [--force]
"""

import argparse
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline

def main():
    parser = argparse.ArgumentParser(description='Inicializar ChromaDB con documentos académicos')
    parser.add_argument('--force', '-f', 
                       action='store_true',
                       help='Forzar recarga de todos los documentos (elimina datos existentes)')
    
    args = parser.parse_args()
    
    print("🗄️ INICIALIZADOR DE CHROMADB")
    print("=" * 40)
    
    try:
        # Crear pipeline RAG
        print("📦 Creando pipeline RAG...")
        rag = create_rag_pipeline(reset_collection=args.force)
        
        # Verificar si ya hay datos
        if not args.force:
            print("🔍 Verificando contenido existente...")
            # Aquí podrías agregar una verificación del contenido existente
            print("💡 Usa --force para recargar todos los documentos")
        
        # Cargar materiales académicos
        print("\n📚 Cargando documentos académicos...")
        print("   📁 Directorio: ./docs")
        print("   📄 Formatos: PDF, TXT, TEX")
        print("   ⚠️  Saltando errores: Sí")
        
        load_result = rag.load_materials(
            data_directory="./docs",
            file_extensions=[".pdf", ".txt", ".tex"],
            skip_on_error=True
        )
        
        # Mostrar resultados
        print("\n" + "=" * 40)
        print("📊 RESULTADOS DE LA CARGA")
        print("=" * 40)
        
        if load_result.get('status') == 'success':
            print(f"✅ ÉXITO: {load_result.get('documents_loaded', 0)} documentos cargados")
            print(f"📝 Chunks procesados: {load_result.get('chunks_created', 0)}")
            print(f"💾 Vector store: {load_result.get('vector_store_path', 'N/A')}")
            
        elif load_result.get('status') in ['partial_error', 'warning']:
            print(f"⚠️  CARGA PARCIAL: {load_result.get('message', 'Algunos documentos fallaron')}")
            print(f"✅ Documentos cargados: {load_result.get('documents_loaded', 0)}")
            print(f"❌ Documentos fallidos: {load_result.get('documents_failed', 0)}")
            print(f"📝 Chunks procesados: {load_result.get('chunks_created', 0)}")
            
        else:
            print(f"❌ ERROR: {load_result.get('message', 'Error desconocido')}")
            print(f"📊 Documentos procesados: {load_result.get('documents_processed', 0)}")
            print(f"📝 Chunks creados: {load_result.get('chunks_created', 0)}")
        
        # Mostrar fuentes cargadas
        if load_result.get('sources'):
            print(f"\n📚 FUENTES CARGADAS ({len(load_result.get('sources', []))}:")
            for i, source in enumerate(load_result.get('sources', [])[:10], 1):  # Mostrar solo las primeras 10
                print(f"   {i:2d}. {source}")
            
            if len(load_result.get('sources', [])) > 10:
                print(f"   ... y {len(load_result.get('sources', [])) - 10} más")
        
        # Mostrar errores si los hay
        if load_result.get('errors'):
            print(f"\n❌ ERRORES ENCONTRADOS ({len(load_result.get('errors', []))}):")
            for i, error in enumerate(load_result.get('errors', []), 1):  # Mostrar todos los errores
                print(f"   {i}. {error}")
        
        print("\n" + "=" * 40)
        if load_result.get('status') == 'success':
            print("🎉 ¡ChromaDB inicializado exitosamente!")
            print("💡 Ahora puedes usar generate_from_config.py para crear ejercicios")
        elif load_result.get('status') in ['partial_error', 'warning']:
            print("⚠️  ChromaDB inicializado parcialmente")
            print("💡 Algunos documentos no se pudieron cargar, pero puedes generar ejercicios")
        else:
            print("❌ Error inicializando ChromaDB")
            print("💡 Revisa los errores y vuelve a intentar")
        
    except KeyboardInterrupt:
        print("\n\n❌ Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
