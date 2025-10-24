#!/usr/bin/env python3
"""
Script para limpiar/vaciar el ChromaDB
Elimina todos los documentos y colecciones
"""

import sys
import shutil
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vector_store import VectorStore

def main():
    print("🧹 LIMPIANDO CHROMADB...")
    print("="*60)
    
    try:
        # Crear instancia del vector store
        print("📦 Conectando a ChromaDB...")
        vector_store = VectorStore()
        
        # Obtener información antes de limpiar
        print("📊 Información actual del ChromaDB:")
        info = vector_store.get_collection_info()
        print(f"   - Documentos: {info.get('document_count', 0)}")
        print(f"   - Colección: {info.get('collection_name', 'N/A')}")
        
        # Limpiar directamente (sin confirmación)
        print(f"\n🗑️  Eliminando todos los documentos del ChromaDB...")
        
        # Limpiar la colección
        success = vector_store.delete_all_documents()
        
        if success:
            print("✅ ChromaDB limpiado exitosamente")
            
            # Verificar que esté vacío
            print("\n📊 Verificando limpieza...")
            info_after = vector_store.get_collection_info()
            print(f"   - Documentos restantes: {info_after.get('document_count', 0)}")
            
            if info_after.get('document_count', 0) == 0:
                print("✅ ChromaDB completamente vacío")
            else:
                print("⚠️  Aún quedan algunos documentos")
        else:
            print("❌ Error al limpiar ChromaDB")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Alternativa: Eliminar manualmente la carpeta data/processed/chroma_db/")

def clean_chroma_directory():
    """
    Función alternativa para limpiar el directorio de ChromaDB
    """
    print("\n🔧 MÉTODO ALTERNATIVO: Limpiar directorio")
    print("="*60)
    
    chroma_dir = Path("data/processed/chroma_db")
    if chroma_dir.exists():
        print(f"📁 Directorio encontrado: {chroma_dir}")
        print("🗑️  Eliminando directorio completo...")
        
        try:
            shutil.rmtree(chroma_dir)
            print("✅ Directorio eliminado exitosamente")
            print("💡 Ahora puedes ejecutar: python initialize_chroma.py")
        except Exception as e:
            print(f"❌ Error eliminando directorio: {str(e)}")
    else:
        print("📁 Directorio no encontrado (ya está limpio)")

if __name__ == "__main__":
    main()
    
    # Mostrar opción alternativa
    print("\n" + "="*60)
    print("💡 OPCIÓN ALTERNATIVA:")
    print("="*60)
    clean_chroma_directory()
