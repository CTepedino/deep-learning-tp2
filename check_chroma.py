#!/usr/bin/env python3
"""
Script para verificar el estado de ChromaDB
Uso: python check_chroma.py
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline

def main():
    print("VERIFICADOR DE CHROMADB")
    print("=" * 40)
    
    try:
        print("Conectando a ChromaDB...")
        rag = create_rag_pipeline()
        
        print("Verificando vector store...")
        vector_store = rag.vector_store.vectorstore
        
        try:
            collection = vector_store._collection
            count = collection.count()
            print(f"Documentos en ChromaDB: {count}")
            
            if count > 0:
                print("ChromaDB esta poblado y listo para usar")
                
                print("\nVerificando fuentes disponibles...")
                try:
                    test_results = vector_store.similarity_search(
                        "probabilidad estadistica", 
                        k=10
                    )
                    
                    if test_results:
                        print(f"Busqueda de prueba exitosa ({len(test_results)} resultados)")
                        print("\nFuentes de ejemplo:")
                        for i, doc in enumerate(test_results[:3], 1):
                            source = doc.metadata.get('source', 'Sin fuente')
                            print(f"   {i}. {Path(source).name}")
                    else:
                        print("No se encontraron resultados en la busqueda de prueba")
                        
                except Exception as e:
                    print(f"Error en busqueda de prueba: {str(e)}")
                    
            else:
                print("ChromaDB esta vacio")
                print("Ejecuta: python initialize_chroma.py")
                
        except Exception as e:
            print(f"Error accediendo a ChromaDB: {str(e)}")
            print("Verifica que ChromaDB este instalado y configurado correctamente")
            
        print("\n" + "=" * 40)
        print("ESTADO FINAL")
        print("=" * 40)
        
        if count > 0:
            print("ChromaDB esta listo para generar ejercicios")
            print("Usa: python generate_from_config.py config_example.json")
        else:
            print("ChromaDB necesita ser inicializado")
            print("Usa: python initialize_chroma.py")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
