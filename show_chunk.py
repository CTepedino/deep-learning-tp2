#!/usr/bin/env python3
"""
Script para mostrar un chunk del ChromaDB y todos sus campos de metadata
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline

def show_chunk_details(chunk, title):
    """Muestra los detalles de un chunk"""
    print(f"\n📄 {title}:")
    print("="*60)
    print(f"📝 Contenido:")
    print(f"   {chunk['content'][:300]}...")
    print(f"\n📊 METADATA COMPLETA:")
    print("="*60)
    
    for key, value in chunk['metadata'].items():
        print(f"   🔹 {key}: {value}")
        
    print(f"\n📈 INFORMACIÓN ADICIONAL:")
    print(f"   - Longitud del contenido: {len(chunk['content'])} caracteres")
    print(f"   - Número de campos de metadata: {len(chunk['metadata'])}")

def main():
    print("🔍 Mostrando chunks del ChromaDB...")
    print("="*60)
    
    try:
        # Crear pipeline RAG
        print("📦 Cargando sistema RAG...")
        rag = create_rag_pipeline()
        
        # Buscar chunk de teoría
        print("🔎 Buscando chunk de TEORÍA...")
        theory_results = rag.search_materials(
            query="probabilidad variables aleatorias",
            k=3
        )
        
        # Buscar chunk de examen
        print("🔎 Buscando chunk de EXAMEN...")
        exam_results = rag.search_materials(
            query="examen probabilidad estadística",
            k=3
        )
        
        if theory_results:
            print(f"\n✅ Se encontraron {len(theory_results)} chunks de teoría")
            show_chunk_details(theory_results[0], "CHUNK DE TEORÍA")
        else:
            print("❌ No se encontraron chunks de teoría")
        
        if exam_results:
            print(f"\n✅ Se encontraron {len(exam_results)} chunks de examen")
            show_chunk_details(exam_results[0], "CHUNK DE EXAMEN")
        else:
            print("❌ No se encontraron chunks de examen")
        
        if not theory_results and not exam_results:
            print("💡 Posiblemente el ChromaDB no está inicializado")
            print("   Ejecuta: python initialize_chroma.py")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
