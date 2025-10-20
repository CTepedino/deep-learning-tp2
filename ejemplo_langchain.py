"""
Ejemplo de uso del sistema RAG con componentes nativos de LangChain
Demuestra el uso de LCEL chains, ChatOpenAI, y retriever nativo
"""

import os
import logging
from dotenv import load_dotenv
from src.rag_pipeline import create_rag_pipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


def ejemplo_basico():
    """Ejemplo básico: generar ejercicios con LCEL chains"""
    print("="*80)
    print("EJEMPLO 1: Generación de ejercicios con LCEL Chains")
    print("="*80)
    
    # Crear pipeline RAG
    rag = create_rag_pipeline()
    
    # Parámetros de consulta
    query_params = {
        "materia": "Probabilidad y estadística",
        "unidad": "Distribución normal",
        "cantidad": 2,
        "nivel_dificultad": "intermedio",
        "tipo_ejercicio": "multiple_choice"
    }
    
    # Generar ejercicios usando LCEL chains
    print("\n🔄 Generando ejercicios con LCEL chains...")
    result = rag.generate_exercises_with_chain(
        query_params=query_params,
        tipo_ejercicio="multiple_choice"
    )
    
    # Mostrar resultados
    if result.get("status") == "error":
        print(f"❌ Error: {result.get('message')}")
        return
    
    print(f"\n✅ Ejercicios generados: {len(result['ejercicios'])}")
    print(f"📊 Método usado: {result['metadata'].get('metodo')}")
    print(f"📚 Chunks recuperados: {result['metadata'].get('chunks_recuperados')}")
    
    # Mostrar primer ejercicio
    if result['ejercicios']:
        ej = result['ejercicios'][0]
        print(f"\n📝 Primer ejercicio:")
        print(f"   Pregunta: {ej['pregunta']}")
        print(f"   Opciones:")
        for i, opcion in enumerate(ej['opciones'], 1):
            print(f"      {chr(64+i)}) {opcion}")
        print(f"   Respuesta correcta: {ej['respuesta_correcta']}")
        print(f"   Pista: {ej['pista']}")


def ejemplo_comparacion():
    """Comparación entre método original y LCEL chains"""
    print("\n" + "="*80)
    print("EJEMPLO 2: Comparación de métodos (Original vs LCEL)")
    print("="*80)
    
    rag = create_rag_pipeline()
    
    query_params = {
        "materia": "Sistemas de Inteligencia Artificial",
        "unidad": "Algoritmos de búsqueda",
        "cantidad": 1,
        "nivel_dificultad": "intermedio"
    }
    
    # Método 1: Original (compatible)
    print("\n🔧 Método Original:")
    result1 = rag.generate_exercises(query_params, k_retrieval=3)
    
    if result1.get("ejercicios"):
        print(f"   ✅ Ejercicios: {len(result1['ejercicios'])}")
        print(f"   📊 Chunks: {result1.get('context_info', {}).get('documents_retrieved', 0)}")
    
    # Método 2: LCEL Chains
    print("\n⚡ Método LCEL:")
    result2 = rag.generate_exercises_with_chain(
        query_params=query_params,
        tipo_ejercicio="multiple_choice"
    )
    
    if result2.get("ejercicios"):
        print(f"   ✅ Ejercicios: {len(result2['ejercicios'])}")
        print(f"   📊 Chunks: {result2['metadata'].get('chunks_recuperados', 0)}")
        print(f"   🎯 Método: {result2['metadata'].get('metodo')}")


def ejemplo_retriever_nativo():
    """Ejemplo de uso del retriever nativo"""
    print("\n" + "="*80)
    print("EJEMPLO 3: Retriever Nativo de LangChain")
    print("="*80)
    
    rag = create_rag_pipeline()
    
    # Obtener retriever nativo
    native_retriever = rag.retriever.get_native_retriever()
    
    print("\n🔍 Usando retriever nativo directamente:")
    print(f"   Tipo: {type(native_retriever)}")
    
    # Invocar retriever
    query = "distribución de probabilidad"
    docs = native_retriever.invoke(query)
    
    print(f"\n📚 Documentos recuperados: {len(docs)}")
    
    if docs:
        print(f"\n📄 Primer documento:")
        print(f"   Contenido: {docs[0].page_content[:150]}...")
        print(f"   Metadata: {docs[0].metadata}")


def ejemplo_chains_directos():
    """Ejemplo de uso directo de los chains del generator"""
    print("\n" + "="*80)
    print("EJEMPLO 4: Uso Directo de Chains del Generator")
    print("="*80)
    
    rag = create_rag_pipeline()
    
    # Acceder a los chains disponibles
    print("\n📦 Chains disponibles:")
    for tipo in rag.generator.chains.keys():
        print(f"   - {tipo}")
    
    # Usar un chain directamente
    chain = rag.generator.chains["multiple_choice"]
    
    # Preparar input
    chain_input = {
        "cantidad": 1,
        "tema": "Variables aleatorias discretas",
        "materia": "Probabilidad y estadística",
        "contexto": """
        Una variable aleatoria discreta puede tomar valores en un conjunto numerable.
        Ejemplos incluyen la distribución binomial y la distribución de Poisson.
        La función de probabilidad P(X=x) da la probabilidad de cada valor posible.
        """,
        "nivel_dificultad": "introductorio"
    }
    
    print("\n⚙️ Invocando chain directamente...")
    result = chain.invoke(chain_input)
    
    print(f"\n✅ Resultado (ya parseado como JSON):")
    print(f"   Ejercicios generados: {len(result.get('ejercicios', []))}")
    
    if result.get('ejercicios'):
        ej = result['ejercicios'][0]
        print(f"\n📝 Ejercicio:")
        print(f"   Pregunta: {ej['pregunta']}")
        print(f"   Opciones: {len(ej['opciones'])} opciones")


def ejemplo_info_sistema():
    """Muestra información del sistema"""
    print("\n" + "="*80)
    print("EJEMPLO 5: Información del Sistema")
    print("="*80)
    
    rag = create_rag_pipeline()
    
    info = rag.get_system_info()
    
    print("\n📊 Configuración del Sistema:")
    print(f"\n   Vector Store:")
    print(f"      - Tipo: {info['vector_store']['type']}")
    print(f"      - Colección: {info['vector_store']['collection_name']}")
    print(f"      - Modelo embeddings: {info['vector_store']['embedding_model']}")
    print(f"      - Documentos: {info['vector_store']['document_count']}")
    
    print(f"\n   Retriever:")
    print(f"      - Tipo: {info['retriever']['type']}")
    print(f"      - k: {info['retriever']['k']}")
    print(f"      - Threshold: {info['retriever']['score_threshold']}")
    
    print(f"\n   Generator:")
    print(f"      - Tipo: {info['generator']['type']}")
    print(f"      - Modelo: {info['generator']['model']}")
    
    print(f"\n   Text Processing:")
    print(f"      - Chunk size: {info['text_processing']['chunk_size']}")
    print(f"      - Chunk overlap: {info['text_processing']['chunk_overlap']}")


def ejemplo_tipos_ejercicios():
    """Genera diferentes tipos de ejercicios"""
    print("\n" + "="*80)
    print("EJEMPLO 6: Diferentes Tipos de Ejercicios")
    print("="*80)
    
    rag = create_rag_pipeline()
    
    tipos = ["multiple_choice", "desarrollo", "practico", "teorico"]
    
    query_params = {
        "materia": "Probabilidad y estadística",
        "unidad": "Esperanza y varianza",
        "cantidad": 1,
        "nivel_dificultad": "intermedio"
    }
    
    for tipo in tipos:
        print(f"\n📚 Generando ejercicio tipo: {tipo}")
        
        result = rag.generate_exercises_with_chain(
            query_params=query_params,
            tipo_ejercicio=tipo
        )
        
        if result.get("ejercicios"):
            ej = result['ejercicios'][0]
            print(f"   ✅ Ejercicio generado")
            print(f"   📝 Pregunta: {ej['pregunta'][:100]}...")
            print(f"   🎯 Campos: {list(ej.keys())}")
        else:
            print(f"   ⚠️ No se pudo generar (posiblemente sin contexto)")


def main():
    """Ejecuta todos los ejemplos"""
    print("\n" + "="*80)
    print("🚀 EJEMPLOS DE USO DE LANGCHAIN EN EL SISTEMA RAG")
    print("="*80)
    
    try:
        # Verificar que existe OPENAI_API_KEY
        if not os.getenv("OPENAI_API_KEY"):
            print("\n❌ Error: OPENAI_API_KEY no encontrada en variables de entorno")
            print("   Por favor, configura tu .env con la API key de OpenAI")
            return
        
        # Ejecutar ejemplos
        ejemplo_basico()
        ejemplo_comparacion()
        ejemplo_retriever_nativo()
        ejemplo_chains_directos()
        ejemplo_info_sistema()
        ejemplo_tipos_ejercicios()
        
        print("\n" + "="*80)
        print("✅ Todos los ejemplos completados exitosamente")
        print("="*80)
        
        print("\n📚 Para más información, consulta LANGCHAIN_USAGE.md")
        
    except Exception as e:
        logger.error(f"Error ejecutando ejemplos: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

