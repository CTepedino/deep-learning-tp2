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
        
        # Buscar chunks por materia y tipo
        print("🔎 Buscando chunks de PROBABILIDAD Y ESTADÍSTICA...")
        prob_theory = rag.search_materials(query="probabilidad variables aleatorias", k=2)
        prob_guide = rag.search_materials(query="guía ejercicios probabilidad", k=2)
        prob_exam = rag.search_materials(query="examen probabilidad estadística", k=2)
        
        print("🔎 Buscando chunks de SISTEMAS DE INTELIGENCIA ARTIFICIAL...")
        ia_theory = rag.search_materials(query="machine learning redes neuronales", k=2)
        ia_guide = rag.search_materials(query="guía ejercicios inteligencia artificial", k=2)
        ia_exam = rag.search_materials(query="examen inteligencia artificial", k=2)
        
        # Mostrar resultados de Probabilidad y Estadística
        print("\n" + "="*60)
        print("📊 PROBABILIDAD Y ESTADÍSTICA")
        print("="*60)
        
        if prob_theory:
            print(f"✅ Teoría: {len(prob_theory)} chunks")
            show_chunk_details(prob_theory[0], "TEORÍA - PROBABILIDAD")
        else:
            print("❌ No se encontraron chunks de teoría")
        
        if prob_guide:
            print(f"✅ Guía: {len(prob_guide)} chunks")
            show_chunk_details(prob_guide[0], "GUÍA - PROBABILIDAD")
        else:
            print("❌ No se encontraron chunks de guía")
        
        if prob_exam:
            print(f"✅ Examen: {len(prob_exam)} chunks")
            show_chunk_details(prob_exam[0], "EXAMEN - PROBABILIDAD")
        else:
            print("❌ No se encontraron chunks de examen")
        
        # Mostrar resultados de Sistemas de IA
        print("\n" + "="*60)
        print("📊 SISTEMAS DE INTELIGENCIA ARTIFICIAL")
        print("="*60)
        
        if ia_theory:
            print(f"✅ Teoría: {len(ia_theory)} chunks")
            show_chunk_details(ia_theory[0], "TEORÍA - IA")
        else:
            print("❌ No se encontraron chunks de teoría")
        
        if ia_guide:
            print(f"✅ Guía: {len(ia_guide)} chunks")
            show_chunk_details(ia_guide[0], "GUÍA - IA")
        else:
            print("❌ No se encontraron chunks de guía")
        
        if ia_exam:
            print(f"✅ Examen: {len(ia_exam)} chunks")
            show_chunk_details(ia_exam[0], "EXAMEN - IA")
        else:
            print("❌ No se encontraron chunks de examen")
        
        # Verificar si no hay nada
        total_results = len(prob_theory) + len(prob_guide) + len(prob_exam) + len(ia_theory) + len(ia_guide) + len(ia_exam)
        if total_results == 0:
            print("\n💡 Posiblemente el ChromaDB no está inicializado")
            print("   Ejecuta: python initialize_chroma.py")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
