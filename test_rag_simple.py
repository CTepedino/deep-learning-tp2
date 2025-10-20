#!/usr/bin/env python3
"""
Script simple para probar el sistema RAG SIN recargar documentos
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import create_rag_pipeline

def main():
    print("\n" + "=" * 70)
    print("🎓 TEST RAG - USANDO BASE DE DATOS EXISTENTE")
    print("=" * 70)
    
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: Configura OPENAI_API_KEY en .env")
        return
    
    print("\n⏳ Inicializando pipeline...")
    rag_pipeline = create_rag_pipeline(reset_collection=False)
    
    system_info = rag_pipeline.get_system_info()
    doc_count = system_info['vector_store']['document_count']
    
    print(f"✅ Pipeline inicializado")
    print(f"📊 Documentos disponibles: {doc_count}")
    
    if doc_count == 0:
        print("\n⚠️ No hay documentos cargados.")
        print("   Ejecuta primero: python test_rag.py para cargar documentos")
        return
    
    # Generar ejercicio
    print("\n" + "=" * 70)
    print("📝 GENERANDO EJERCICIO DE PRUEBA")
    print("=" * 70)
    
    query_params = {
        "materia": "Probabilidad y estadística",
        "unidad": "Variables Aleatorias",
        "cantidad": 1,
        "nivel_dificultad": "intermedio",
        "tipo_ejercicio": "multiple_choice"
    }
    
    print(f"\n📋 Parámetros:")
    for key, value in query_params.items():
        print(f"   - {key}: {value}")
    
    print("\n⏳ Generando con LangChain (ChatOpenAI + LCEL)...")
    result = rag_pipeline.generate_exercises(
        query_params=query_params,
        k_retrieval=5
    )
    
    if result.get('status') == 'error':
        print(f"❌ Error: {result.get('message')}")
        return
    
    # Mostrar ejercicio
    print("\n" + "=" * 70)
    print("✨ EJERCICIO GENERADO")
    print("=" * 70)
    
    if result.get('ejercicios'):
        exercise = result['ejercicios'][0]
        print(f"\n📝 {exercise['pregunta']}\n")
        print("Opciones:")
        for i, opcion in enumerate(exercise['opciones'], 1):
            letra = chr(64 + i)  # A, B, C, D
            print(f"  {letra}) {opcion}")
        
        print(f"\n✅ Respuesta correcta: {exercise['respuesta_correcta']}")
        print(f"💡 Pista: {exercise.get('pista', 'N/A')}")
        
        # Metadata
        metadata = result.get('metadata', {})
        print(f"\n📊 Metadata:")
        print(f"   - Modelo: {metadata.get('modelo_usado')}")
        print(f"   - Chunks recuperados: {metadata.get('chunks_recuperados')}")
        print(f"   - Fuentes: {len(metadata.get('fuentes', []))} documentos")
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETADO - LANGCHAIN FUNCIONA CORRECTAMENTE")
        print("=" * 70)
    else:
        print("⚠️ No se generaron ejercicios")

if __name__ == "__main__":
    main()

